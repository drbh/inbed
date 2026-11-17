use inbed::api::Api;
use inbed::backend::{Backend, Batch};
use inbed::engine::{Device, Engine, EngineConfig};
use inbed::model::{self, Spec};
use std::sync::Arc;
use std::time::Instant;

const USAGE: &str = "\
inbed: a small, dependency-free text embedding server (OpenAI-compatible /v1/embeddings)

USAGE:
  inbed serve    --model <M> [--model <M>...] [--addr 127.0.0.1:8070] [--device auto|metal|cpu]
                 [--threads 32] [--max-batch-tokens 16384] [--queue 64] [--max-inputs 2048]
  inbed embed    --model <M> [--device D] [--query] [--dims N] <text>...
  inbed tokenize --model <M> [--query] <text>...
  inbed bench    --model <M> [--device D] [--batches 1,8,32,128] [--seq-len 128] [--iters 10]
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
        const BOOLS: &[&str] = &["query", "help"];
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
    let cfg = EngineConfig {
        device: device(a)?,
        max_batch_tokens: a.num("max-batch-tokens", 16384)?,
        queue: a.num("queue", 64)?,
    };
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
    let addr = a.get("addr").unwrap_or_else(|| "127.0.0.1:8070".into());
    let listener = std::net::TcpListener::bind(&addr).map_err(|e| format!("bind {addr}: {e}"))?;
    eprintln!(
        "inbed: listening on http://{addr} (device {:?})",
        cfg.device
    );
    let api = Arc::new(Api {
        engines,
        max_inputs: a.num("max-inputs", 2048)?,
    });
    inbed::http::serve(
        listener,
        a.num("threads", 32)?,
        Arc::new(move |r| api.handle(r)),
    );
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

fn bench(a: &Args) -> Result<(), String> {
    let spec = one_model(a)?;
    let seq_len = a.num("seq-len", 128)?.min(spec.max_tokens);
    let iters = a.num("iters", 10)?.max(1);
    let batches: Vec<usize> = a
        .get("batches")
        .unwrap_or_else(|| "1,8,32,128".into())
        .split(',')
        .map(|s| {
            s.trim()
                .parse()
                .map_err(|_| format!("bad --batches entry {s}"))
        })
        .collect::<Result<_, _>>()?;
    let tok = spec.tokenizer()?;
    let text = "The quick brown fox jumps over the lazy dog while embeddings map text to vectors. "
        .repeat(seq_len);
    let doc = tok.encode(&text, seq_len);
    let short = tok.encode(
        &format!("{}what is a fast vector database?", spec.query_prefix),
        spec.max_tokens,
    );
    match device(a)? {
        #[cfg(target_os = "macos")]
        Device::Metal => {
            let mut b = inbed::metal::Metal::new()?;
            eprintln!("device: {}", b.device_name());
            run_bench(&spec, &mut b, &doc, &short, &batches, iters)?;
            for (k, n, t) in b.take_profile() {
                println!("  {k:<14} {n:>7} calls {:>10.2} ms", t * 1e3);
            }
            Ok(())
        }
        #[cfg(not(target_os = "macos"))]
        Device::Metal => Err("Metal is only available on macOS".into()),
        Device::Cpu => run_bench(
            &spec,
            &mut inbed::cpu::Cpu::new(),
            &doc,
            &short,
            &batches,
            iters,
        ),
    }
}

fn run_bench<B: Backend>(
    spec: &Spec,
    b: &mut B,
    doc: &[u32],
    short: &[u32],
    batches: &[usize],
    iters: usize,
) -> Result<(), String> {
    let t0 = Instant::now();
    let w = model::load_weights(spec, b)?;
    println!(
        "model {} on {}: loaded in {:.2}s",
        spec.id,
        b.name(),
        t0.elapsed().as_secs_f64()
    );
    // warm-up
    model::forward(spec, &w, b, &Batch::new(&[short]));
    let mut lat: Vec<f64> = (0..iters.max(20))
        .map(|_| {
            let t = Instant::now();
            model::forward(spec, &w, b, &Batch::new(&[short]));
            t.elapsed().as_secs_f64() * 1e3
        })
        .collect();
    lat.sort_by(|a, b| a.partial_cmp(b).unwrap());
    println!(
        "query latency ({} tokens, batch 1): p50 {:.2} ms, p90 {:.2} ms",
        short.len(),
        lat[lat.len() / 2],
        lat[lat.len() * 9 / 10]
    );
    println!(
        "{:>6} {:>8} {:>10} {:>12}",
        "batch", "tokens", "ms/batch", "tokens/s"
    );
    for &bs in batches {
        let seqs: Vec<&[u32]> = (0..bs).map(|_| doc).collect();
        let batch = Batch::new(&seqs);
        model::forward(spec, &w, b, &batch);
        let t = Instant::now();
        for _ in 0..iters {
            model::forward(spec, &w, b, &batch);
        }
        let ms = t.elapsed().as_secs_f64() * 1e3 / iters as f64;
        println!(
            "{:>6} {:>8} {:>10.2} {:>12.0}",
            bs,
            batch.tokens(),
            ms,
            batch.tokens() as f64 / ms * 1e3
        );
    }
    Ok(())
}
