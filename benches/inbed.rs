//! The inbed benchmark suite: `cargo bench`, then compare runs.
//!
//! ```text
//! cargo bench                                   # every cached registry model, every suite
//! cargo bench -- --quick                        # fewer rounds, no long documents (~1 min)
//! cargo bench -- --models all-MiniLM-L6-v2 --suites model,http
//! cargo bench -- --save-baseline main           # remember this run as "main"
//! cargo bench -- --baseline main                # compare against it (exit 1 on a regression
//!                                               #   with --fail-on-regression)
//! ```
//!
//! Suites: `model` (raw forward passes: load, query latency, throughput, long documents),
//! `idle` (query latency after idle gaps), `load` (query latency under bulk ingest), `http`
//! (round trips over TCP and a Unix socket, JSON vs binary).
//!
//! Every run is written to `target/inbed-bench/runs/<date>.json` and `target/inbed-bench/latest.json`
//! with its metadata (git revision, CPU/GPU, model revisions, load average, INBED_* env).
//! Baselines live in `target/inbed-bench/baselines/<name>.json`. Results only compare between runs
//! on the same machine; the run warns when other processes are busy.

use inbed::bench::{self, Load, Opts, Recorder};
use inbed::engine::{Device, EngineConfig};
use inbed::model::Spec;
use std::path::PathBuf;
use std::time::Duration;

struct Args {
    models: Vec<String>,
    suites: Vec<String>,
    device: Device,
    opts: Opts,
    save: Option<String>,
    baseline: Option<String>,
    threshold: f64,
    fail: bool,
    out: PathBuf,
    cooldown: Duration,
}

const SUITES: &[&str] = &["model", "idle", "load", "http"];

fn parse() -> Result<Args, String> {
    let mut a = Args {
        models: vec![],
        suites: SUITES.iter().map(|s| s.to_string()).collect(),
        device: Device::auto(),
        opts: Opts::default(),
        save: None,
        baseline: None,
        threshold: 0.05,
        fail: false,
        out: PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("target/inbed-bench"),
        cooldown: Duration::from_secs(2),
    };
    let mut it = std::env::args().skip(1);
    let list = |v: String| -> Vec<String> {
        v.split(',')
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
            .collect()
    };
    while let Some(k) = it.next() {
        let mut val = || it.next().ok_or(format!("{k} needs a value"));
        let num = |v: String| v.parse::<f64>().map_err(|_| format!("{k}: not a number"));
        match k.as_str() {
            "--bench" => {} // passed by cargo
            "--quick" => {
                a.opts.quick = true;
                a.opts.rounds = 3;
                a.opts.iters = 15;
                a.opts.seconds = 2.0;
            }
            "--fail-on-regression" => a.fail = true,
            "--models" => a.models = list(val()?),
            "--suites" => a.suites = list(val()?),
            "--device" => a.device = Device::parse(&val()?)?,
            "--rounds" => a.opts.rounds = num(val()?)?.max(1.0) as usize,
            "--iters" => a.opts.iters = num(val()?)?.max(2.0) as usize,
            "--seconds" => a.opts.seconds = num(val()?)?,
            "--save-baseline" => a.save = Some(val()?),
            "--baseline" => a.baseline = Some(val()?),
            "--threshold" => a.threshold = num(val()?)? / 100.0,
            "--out" => a.out = PathBuf::from(val()?),
            "--cooldown-s" => a.cooldown = Duration::from_secs_f64(num(val()?)?),
            "-h" | "--help" => {
                println!(
                    "cargo bench -- [--quick] [--models a,b] [--suites {}] [--device metal|cuda|cpu]\n\
                     \t[--rounds 5] [--iters 30] [--seconds 4] [--cooldown-s 2]\n\
                     \t[--save-baseline NAME] [--baseline NAME] [--threshold 5 (%)] [--fail-on-regression]\n\
                     \t[--out target/inbed-bench]",
                    SUITES.join(",")
                );
                std::process::exit(0);
            }
            _ => return Err(format!("unknown argument {k}")),
        }
    }
    if let Some(s) = a.suites.iter().find(|s| !SUITES.contains(&s.as_str())) {
        return Err(format!("unknown suite {s} ({})", SUITES.join(", ")));
    }
    Ok(a)
}

/// Models to run: the ones asked for, or every cached registry model.
fn specs(a: &Args) -> Result<Vec<Spec>, String> {
    let names: Vec<String> = if a.models.is_empty() {
        inbed::hub::REGISTRY
            .iter()
            .filter(|k| inbed::hub::cached_repo(k.repo).is_some())
            .map(|k| k.id.to_string())
            .collect()
    } else {
        a.models.clone()
    };
    names
        .iter()
        .map(|n| {
            // accept a short name such as all-MiniLM-L6-v2
            let full = inbed::hub::REGISTRY
                .iter()
                .find(|k| k.id.ends_with(&format!("/{n}")))
                .map(|k| k.id)
                .unwrap_or(n);
            let r = inbed::hub::resolve(full)?;
            Spec::load(&r.id, &r.dir, r.allowed_dims.as_deref())
        })
        .collect()
}

fn run(a: &Args) -> Result<usize, String> {
    let specs = specs(a)?;
    if specs.is_empty() {
        return Err(
            "no models cached; `hf download sentence-transformers/all-MiniLM-L6-v2`".into(),
        );
    }
    if let Some(l) = bench::load_avg() {
        eprintln!("load average (1m): {l:.2}");
    }
    let busy = bench::busy_processes(25.0);
    if !busy.is_empty() {
        eprintln!("warning: other processes are busy; results will be noisy:");
        for (c, n) in &busy {
            eprintln!("  {c:>6.1}%  {n}");
        }
    }
    let mut r = Recorder::new(true);
    let cfg = EngineConfig {
        device: a.device,
        ..Default::default()
    };
    for spec in &specs {
        for suite in &a.suites {
            eprintln!("== {} / {suite}", spec.id);
            match suite.as_str() {
                "model" => {
                    let batches: &[usize] = if a.opts.quick {
                        &[1, 32]
                    } else {
                        &[1, 8, 32, 128]
                    };
                    bench::model_suite_on(spec, a.device, a.opts, batches, 128, &mut r)?
                }
                "idle" => {
                    let gaps: &[u64] = if a.opts.quick {
                        &[0, 50]
                    } else {
                        &[0, 5, 20, 50, 100]
                    };
                    bench::idle_suite(spec, cfg, gaps, a.opts, &mut r)?
                }
                "load" => bench::load_suite(spec, cfg, Load::default(), a.opts, &mut r)?,
                "http" => bench::http_suite(spec, cfg, a.opts, &mut r)?,
                _ => unreachable!(),
            }
            std::thread::sleep(a.cooldown);
        }
    }

    let argv: Vec<String> = std::env::args()
        .skip(1)
        .filter(|a| a != "--bench")
        .collect();
    let refs: Vec<&Spec> = specs.iter().collect();
    let json = bench::to_json(bench::meta(&refs, a.device, a.opts, &argv), &r.metrics).to_string();
    let runs = a.out.join("runs");
    std::fs::create_dir_all(&runs).map_err(|e| e.to_string())?;
    let date = json
        .split("\"date\":\"")
        .nth(1)
        .and_then(|s| s.get(..20))
        .unwrap_or("run")
        .replace(':', "-");
    let file = runs.join(format!("{date}.json"));
    std::fs::write(&file, &json).map_err(|e| e.to_string())?;
    std::fs::write(a.out.join("latest.json"), &json).map_err(|e| e.to_string())?;
    eprintln!("\nwrote {}", file.display());

    println!();
    for m in &r.metrics {
        println!("{}", bench::fmt_metric(m));
    }
    let bdir = a.out.join("baselines");
    if let Some(name) = &a.save {
        std::fs::create_dir_all(&bdir).map_err(|e| e.to_string())?;
        let p = bdir.join(format!("{name}.json"));
        std::fs::write(&p, &json).map_err(|e| e.to_string())?;
        eprintln!("saved baseline {}", p.display());
    }
    let mut worse = 0;
    if let Some(name) = &a.baseline {
        let p = bdir.join(format!("{name}.json"));
        let text = std::fs::read_to_string(&p).map_err(|e| format!("{}: {e}", p.display()))?;
        let base = inbed::json::Json::parse(&text)?;
        if let Some(m) = base.get("meta") {
            println!(
                "\nbaseline {name}: git {} on {}, {}",
                m.get("git").and_then(|d| d.as_str()).unwrap_or("?"),
                m.get("date").and_then(|d| d.as_str()).unwrap_or("?"),
                m.get("cpu").and_then(|d| d.as_str()).unwrap_or("?"),
            );
        }
        println!(
            "(changes count when > {:.0}% and outside both runs' spread)\n",
            a.threshold * 100.0
        );
        worse = bench::compare(&bench::metrics_from_json(&base)?, &r.metrics, a.threshold);
        println!("\n{worse} regression(s)");
    }
    Ok(worse)
}

fn main() {
    let a = match parse() {
        Ok(a) => a,
        Err(e) => {
            eprintln!("bench: {e}");
            std::process::exit(2);
        }
    };
    match run(&a) {
        Ok(worse) if worse > 0 && a.fail => std::process::exit(1),
        Ok(_) => {}
        Err(e) => {
            eprintln!("bench: {e}");
            std::process::exit(1);
        }
    }
}
