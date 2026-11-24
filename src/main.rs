use inbed::api::Api;
use inbed::engine::{Device, Engine, EngineConfig};
use inbed::model::{self, Spec};
use std::sync::Arc;

const USAGE: &str = "\
inbed: a small, dependency-free text embedding server (OpenAI-compatible /v1/embeddings)

USAGE:
  inbed serve    --model <M> [--model <M>...] [--addr 127.0.0.1:8070] [--unix PATH [--unix-mode 660]]
                 [--device auto|metal|cuda|cpu] [--threads 32] [--idle-timeout-s 600]
                 [--max-batch-tokens 16384] [--queue 64] [--max-inputs 2048]
                 [--query-max-wait-ms 100] [--doc-max-wait-ms 5000] [--doc-slice-ms 20] [--spin-us 50]
  inbed embed    --model <M> [--device D] [--query] [--dims N] <text>...
  inbed tokenize --model <M> [--query] <text>...
  inbed bench    --model <M> [--mode model|idle|load|http] [--device D] [--rounds 3] [--iters 30]
                 model: [--batches 1,8,32,128] [--seq-len 128] [--long]
                 idle:  [--gaps 0,1,5,10,20,50,100]
                 load:  [--seconds 4] [--doc-clients 4] [--doc-batch 32] [--query-every-ms 10]
                 (engine flags as for serve apply to idle, load and http)
  Reproducible suite with saved JSON and baselines: cargo bench -- --help
  inbed models

<M> is a registry name (qwen/qwen3-embedding-0p6b), a Hugging Face repo id in the local cache,
a snapshot directory, or <id>=<repo-or-dir>. Download weights with `hf download <repo>`.
Environment: INBED_<FLAG> (e.g. INBED_ADDR, INBED_MODEL=a,b, INBED_DEVICE) sets defaults.
";

struct Args {
    flags: Vec<(String, String)>,
    bools: Vec<String>,
    rest: Vec<String>,
}

impl Args {
    fn parse(v: &[String]) -> Result<Args, String> {
        const BOOLS: &[&str] = &["query", "help", "long"];
        let mut a = Args {
            flags: vec![],
            bools: vec![],
            rest: vec![],
        };
        let mut i = 0;
        while i < v.len() {
            let s = &v[i];
            if let Some(name) = s.strip_prefix("--") {
                if let Some((k, val)) = name.split_once('=') {
                    a.flags.push((k.to_string(), val.to_string()));
                } else if BOOLS.contains(&name) {
                    a.bools.push(name.to_string());
                } else {
                    let val = v
                        .get(i + 1)
                        .ok_or_else(|| format!("--{name} needs a value"))?;
                    a.flags.push((name.to_string(), val.clone()));
                    i += 1;
                }
            } else if s == "-h" {
                a.bools.push("help".into());
            } else {
                a.rest.push(s.clone());
            }
            i += 1;
        }
        Ok(a)
    }
    fn all(&self, k: &str) -> Vec<String> {
        let v: Vec<String> = self
            .flags
            .iter()
            .filter(|(n, _)| n == k)
            .map(|(_, v)| v.clone())
            .collect();
        if !v.is_empty() {
            return v;
        }
        env(k)
            .map(|e| {
                e.split(',')
                    .map(|s| s.trim().to_string())
                    .filter(|s| !s.is_empty())
                    .collect()
            })
            .unwrap_or_default()
    }
    fn get(&self, k: &str) -> Option<String> {
        self.flags
            .iter()
            .rev()
            .find(|(n, _)| n == k)
            .map(|(_, v)| v.clone())
            .or_else(|| env(k))
    }
    fn num(&self, k: &str, d: usize) -> Result<usize, String> {
        self.get(k)
            .map(|v| v.parse().map_err(|_| format!("--{k}: not a number: {v}")))
            .unwrap_or(Ok(d))
    }
    fn has(&self, k: &str) -> bool {
        self.bools.iter().any(|b| b == k)
    }
}

fn env(k: &str) -> Option<String> {
    std::env::var(format!("INBED_{}", k.to_uppercase().replace('-', "_")))
        .ok()
        .filter(|v| !v.is_empty())
}

fn spec_for(arg: &str) -> Result<Spec, String> {
    let r = inbed::hub::resolve(arg)?;
    Spec::load(&r.id, &r.dir, r.allowed_dims.as_deref())
}

fn one_model(a: &Args) -> Result<Spec, String> {
    let m = a.all("model");
    match m.as_slice() {
        [one] => spec_for(one),
        [] => Err("--model is required".into()),
        _ => Err("exactly one --model".into()),
    }
}

fn device(a: &Args) -> Result<Device, String> {
    a.get("device")
        .map(|d| Device::parse(&d))
        .unwrap_or(Ok(Device::auto()))
}

fn main() {
    let argv: Vec<String> = std::env::args().skip(1).collect();
    let (cmd, rest) = match argv.split_first() {
        Some((c, r)) => (c.as_str(), r),
        None => {
            eprint!("{USAGE}");
            std::process::exit(2);
        }
    };
    let r = Args::parse(rest).and_then(|a| {
        if a.has("help") {
            print!("{USAGE}");
            return Ok(());
        }
        match cmd {
            "serve" => serve(&a),
            "embed" => embed(&a),
            "tokenize" => tokenize(&a),
            "bench" => bench(&a),
            "models" => models(),
            "help" | "--help" | "-h" => {
                print!("{USAGE}");
                Ok(())
            }
            c => Err(format!("unknown command {c:?}\n\n{USAGE}")),
        }
    });
    if let Err(e) = r {
        eprintln!("inbed: {e}");
        std::process::exit(1);
    }
}

fn serve(a: &Args) -> Result<(), String> {
    let models = a.all("model");
    if models.is_empty() {
        return Err("serve: at least one --model is required".into());
    }
    let cfg = engine_cfg(a)?;
    let mut engines = Vec::new();
    for m in &models {
        let spec = spec_for(m)?;
        if engines.iter().any(|e: &Arc<Engine>| e.spec.id == spec.id) {
            return Err(format!("model id {} given twice", spec.id));
        }
        eprintln!(
            "inbed: loading {} ({}, {} dims, max {} tokens) from {}",
            spec.id,
            spec.arch.name(),
            spec.dims,
            spec.max_tokens,
            spec.dir.display()
        );
        engines.push(Engine::start(spec, cfg)?);
    }
    let api = Arc::new(Api {
        engines,
        max_inputs: a.num("max-inputs", 2048)?,
    });
    let handler: inbed::http::Handler = Arc::new(move |r| api.handle(r));
    let idle = a.num("idle-timeout-s", 600)?;
    let hcfg = inbed::http::Config {
        threads: a.num("threads", 32)?,
        idle_timeout: (idle > 0).then(|| std::time::Duration::from_secs(idle as u64)),
    };
    // one connection limit across both listeners
    let active = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let unix = a.get("unix");
    // TCP unless only --unix was given
    if a.get("addr").is_some() || unix.is_none() {
        let addr = a.get("addr").unwrap_or_else(|| "127.0.0.1:8070".into());
        let l = std::net::TcpListener::bind(&addr).map_err(|e| format!("bind {addr}: {e}"))?;
        eprintln!(
            "inbed: listening on http://{addr} (device {:?})",
            cfg.device
        );
        inbed::http::spawn(l, hcfg, active.clone(), handler.clone());
    }
    if let Some(path) = unix {
        let mode = u32::from_str_radix(&a.get("unix-mode").unwrap_or_else(|| "660".into()), 8)
            .map_err(|_| "--unix-mode: not an octal mode".to_string())?;
        let l = inbed::http::bind_unix(&path, mode)?;
        eprintln!(
            "inbed: listening on unix://{path} (device {:?})",
            cfg.device
        );
        inbed::http::spawn(l, hcfg, active, handler);
    }
    loop {
        std::thread::park();
    }
}

fn embed(a: &Args) -> Result<(), String> {
    let spec = one_model(a)?;
    let dims = a.num("dims", spec.dims)?;
    if !spec.allowed_dims.contains(&dims) {
        return Err(format!("--dims must be one of {:?}", spec.allowed_dims));
    }
    let e = Engine::start(
        spec,
        EngineConfig {
            device: device(a)?,
            ..Default::default()
        },
    )?;
    e.wait_ready()?;
    let seqs: Vec<Vec<u32>> = a
        .rest
        .iter()
        .map(|t| e.tokenize(t, a.has("query")))
        .collect();
    for v in e.embed(seqs)? {
        let v = model::truncate_dims(v, dims);
        let mut s = String::from("[");
        for (i, x) in v.iter().enumerate() {
            if i > 0 {
                s.push(',');
            }
            inbed::json::write_f32(&mut s, *x);
        }
        println!("{s}]");
    }
    Ok(())
}

fn tokenize(a: &Args) -> Result<(), String> {
    let spec = one_model(a)?;
    let tok = spec.tokenizer()?;
    for t in &a.rest {
        let prefix = if a.has("query") {
            &spec.query_prefix
        } else {
            &spec.document_prefix
        };
        let ids = tok.encode(&format!("{prefix}{t}"), spec.max_tokens);
        println!("{ids:?}");
    }
    Ok(())
}

fn models() -> Result<(), String> {
    println!("registry (served under these ids):");
    for k in inbed::hub::REGISTRY {
        let cached = inbed::hub::cached_repo(k.repo)
            .map(|p| p.display().to_string())
            .unwrap_or_else(|| format!("not cached: hf download {}", k.repo));
        println!("  {:<42} {:<40} {}", k.id, k.repo, cached);
    }
    Ok(())
}

fn engine_cfg(a: &Args) -> Result<EngineConfig, String> {
    let d = EngineConfig::default();
    Ok(EngineConfig {
        device: device(a)?,
        max_batch_tokens: a.num("max-batch-tokens", d.max_batch_tokens)?,
        queue: a.num("queue", d.queue)?,
        query_max_wait_ms: a.num("query-max-wait-ms", d.query_max_wait_ms as usize)? as u64,
        doc_max_wait_ms: a.num("doc-max-wait-ms", d.doc_max_wait_ms as usize)? as u64,
        doc_slice_ms: a.num("doc-slice-ms", d.doc_slice_ms as usize)? as u64,
        spin_us: a.num("spin-us", d.spin_us as usize)? as u64,
    })
}

fn bench(a: &Args) -> Result<(), String> {
    use inbed::bench::{self, Load, Opts, Recorder};
    let spec = one_model(a)?;
    let d = Opts::default();
    let opts = Opts {
        rounds: a.num("rounds", 3)?.max(1),
        iters: a.num("iters", d.iters)?.max(2),
        seconds: a.num("seconds", 4)? as f64,
        quick: true,
    };
    let list = |k: &str, def: &str| -> Result<Vec<usize>, String> {
        a.get(k)
            .unwrap_or_else(|| def.into())
            .split(',')
            .map(|s| s.trim().parse().map_err(|_| format!("bad --{k} entry {s}")))
            .collect()
    };
    let mut r = Recorder::new(false);
    let cfg = engine_cfg(a)?;
    match a.get("mode").as_deref().unwrap_or("model") {
        "model" => bench::model_suite_on(
            &spec,
            cfg.device,
            Opts {
                quick: !a.has("long"),
                ..opts
            },
            &list("batches", "1,8,32,128")?,
            a.num("seq-len", 128)?,
            &mut r,
        )?,
        "idle" => {
            let gaps: Vec<u64> = list("gaps", "0,1,5,10,20,50,100")?
                .into_iter()
                .map(|g| g as u64)
                .collect();
            bench::idle_suite(&spec, cfg, &gaps, opts, &mut r)?
        }
        "load" => bench::load_suite(
            &spec,
            cfg,
            Load {
                clients: a.num("doc-clients", 4)?,
                doc_batch: a.num("doc-batch", 32)?,
                seq_len: a.num("seq-len", 128)?,
                query_every: std::time::Duration::from_millis(a.num("query-every-ms", 10)? as u64),
            },
            opts,
            &mut r,
        )?,
        "http" => bench::http_suite(&spec, cfg, opts, &mut r)?,
        m => return Err(format!("unknown --mode {m} (model, idle, load, http)")),
    }
    for m in &r.metrics {
        println!("{}", bench::fmt_metric(m));
    }
    Ok(())
}
