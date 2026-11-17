//! Benchmarks shared by `inbed bench` and `cargo bench`.
//!
//! Every measurement is repeated over several rounds. A metric reports the median of its round
//! values, with the lowest and highest round as its spread, so two runs can be compared. A run
//! records where it ran (`meta`) and saves as JSON; `compare` diffs two runs and flags only
//! changes larger than both a threshold and the spread.

use crate::api::Api;
use crate::backend::{Backend, Batch};
use crate::engine::{Device, Engine, EngineConfig, Kind, SubmitError};
use crate::json::Json;
use crate::model::{self, Spec};
use std::io::{BufRead, BufReader, Read, Write};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering::Relaxed};
use std::time::{Duration, Instant};

#[derive(Clone, Debug)]
pub struct Metric {
    /// `<model>/<device>/<suite>/<what>`
    pub name: String,
    pub unit: &'static str,
    pub value: f64,
    pub lo: f64,
    pub hi: f64,
    pub lower_is_better: bool,
}

#[derive(Clone, Copy, Debug)]
pub struct Opts {
    /// repetitions of every measurement; the metric is the median round
    pub rounds: usize,
    /// samples per round for latency measurements
    pub iters: usize,
    /// seconds of query traffic per round in the load benchmark
    pub seconds: f64,
    /// skip the slow parts (long documents, most gaps)
    pub quick: bool,
}

impl Default for Opts {
    fn default() -> Self {
        Opts {
            rounds: 5,
            iters: 30,
            seconds: 4.0,
            quick: false,
        }
    }
}

/// Fixed inputs, so every run embeds exactly the same tokens.
pub struct Inputs {
    pub query: Vec<u32>,
    pub doc: Vec<u32>,
}

pub const QUERY_TEXT: &str = "what is a fast vector database?";
const DOC_TEXT: &str =
    "The quick brown fox jumps over the lazy dog while embeddings map text to vectors. ";

impl Inputs {
    pub fn new(spec: &Spec, doc_len: usize) -> Result<Inputs, String> {
        let tok = spec.tokenizer()?;
        Ok(Inputs {
            query: tok.encode(
                &format!("{}{QUERY_TEXT}", spec.query_prefix),
                spec.max_tokens,
            ),
            doc: tok.encode(&DOC_TEXT.repeat(doc_len), doc_len.min(spec.max_tokens)),
        })
    }
    /// A document of exactly `n` tokens (truncated repetition of the fixed text).
    pub fn doc_of(spec: &Spec, n: usize) -> Result<Vec<u32>, String> {
        Ok(spec.tokenizer()?.encode(&DOC_TEXT.repeat(n), n))
    }
}

pub fn pct(v: &[f64], p: f64) -> f64 {
    if v.is_empty() {
        return f64::NAN;
    }
    let mut v = v.to_vec();
    v.sort_by(|a, b| a.total_cmp(b));
    v[((v.len() as f64 * p) as usize).min(v.len() - 1)]
}

fn short_name(id: &str) -> &str {
    id.rsplit('/').next().unwrap_or(id)
}

pub struct Recorder {
    prefix: String,
    pub metrics: Vec<Metric>,
    verbose: bool,
}

impl Recorder {
    pub fn new(verbose: bool) -> Recorder {
        Recorder {
            prefix: String::new(),
            metrics: vec![],
            verbose,
        }
    }
    pub fn scope(&mut self, spec: &Spec, device: Device, suite: &str) {
        self.prefix = format!(
            "{}/{}/{suite}",
            short_name(&spec.id),
            format!("{device:?}").to_lowercase()
        );
    }
    /// One metric from per-round values.
    pub fn add(&mut self, what: &str, unit: &'static str, rounds: &[f64], lower_is_better: bool) {
        let m = Metric {
            name: format!("{}/{what}", self.prefix),
            unit,
            value: pct(rounds, 0.5),
            lo: rounds.iter().copied().fold(f64::INFINITY, f64::min),
            hi: rounds.iter().copied().fold(f64::NEG_INFINITY, f64::max),
            lower_is_better,
        };
        if self.verbose {
            eprintln!("  {}", fmt_metric(&m));
        }
        self.metrics.push(m);
    }
}

pub fn fmt_metric(m: &Metric) -> String {
    format!(
        "{:<58} {:>11} {:<6} [{} – {}]",
        m.name,
        num(m.value),
        m.unit,
        num(m.lo),
        num(m.hi)
    )
}

fn num(x: f64) -> String {
    if x.abs() >= 1000.0 {
        format!("{x:.0}")
    } else if x.abs() >= 10.0 {
        format!("{x:.1}")
    } else {
        format!("{x:.3}")
    }
}

// ------------------------------------------------------------------ model: raw forward passes

/// Load time, batch-1 query latency, throughput by batch size, long documents. Calls the backend
/// directly, without the engine or HTTP.
#[allow(clippy::too_many_arguments)]
pub fn model_suite<B: Backend>(
    spec: &Spec,
    device: Device,
    b: &mut B,
    opts: Opts,
    batches: &[usize],
    seq_len: usize,
    r: &mut Recorder,
) -> Result<(), String> {
    r.scope(spec, device, "model");
    let inp = Inputs::new(spec, seq_len)?;
    // load time over a few loads (weights are dropped between them)
    let mut loads = vec![];
    for _ in 0..opts.rounds.min(3) - 1 {
        let t0 = Instant::now();
        drop(model::load_weights(spec, b)?);
        loads.push(t0.elapsed().as_secs_f64());
    }
    let t0 = Instant::now();
    let w = model::load_weights(spec, b)?;
    loads.push(t0.elapsed().as_secs_f64());
    r.add("load", "s", &loads, true);
    let fwd = |b: &mut B, seqs: &[&[u32]]| {
        let t = Instant::now();
        model::forward(spec, &w, b, &Batch::new(seqs));
        t.elapsed().as_secs_f64()
    };
    for _ in 0..3 {
        fwd(b, &[&inp.query]);
    }
    let (mut p50, mut p90) = (vec![], vec![]);
    for _ in 0..opts.rounds {
        let s: Vec<f64> = (0..opts.iters)
            .map(|_| fwd(b, &[&inp.query]) * 1e3)
            .collect();
        p50.push(pct(&s, 0.5));
        p90.push(pct(&s, 0.9));
    }
    let qt = inp.query.len();
    r.add(&format!("query_{qt}tok_p50"), "ms", &p50, true);
    r.add(&format!("query_{qt}tok_p90"), "ms", &p90, true);
    let iters = (opts.iters / 5).max(2);
    for &bs in batches {
        let seqs: Vec<&[u32]> = (0..bs).map(|_| &inp.doc[..]).collect();
        let tokens = (bs * inp.doc.len()) as f64;
        fwd(b, &seqs);
        let tps: Vec<f64> = (0..opts.rounds)
            .map(|_| {
                let t: f64 = (0..iters).map(|_| fwd(b, &seqs)).sum();
                tokens * iters as f64 / t
            })
            .collect();
        r.add(
            &format!("throughput_b{bs}x{}", inp.doc.len()),
            "tok/s",
            &tps,
            false,
        );
    }
    if !opts.quick {
        for n in [1024, 4096, 8192] {
            if n > spec.max_tokens {
                continue;
            }
            let d = Inputs::doc_of(spec, n)?;
            fwd(b, &[&d]);
            let rounds = if n >= 4096 { 2 } else { opts.rounds.min(3) };
            let s: Vec<f64> = (0..rounds).map(|_| fwd(b, &[&d])).collect();
            r.add(&format!("long_{n}tok"), "s", &s, true);
        }
    }
    Ok(())
}

/// `model_suite` on the requested device. With `INBED_PROFILE=1` on Metal, prints GPU time per kernel.
pub fn model_suite_on(
    spec: &Spec,
    device: Device,
    opts: Opts,
    batches: &[usize],
    seq_len: usize,
    r: &mut Recorder,
) -> Result<(), String> {
    match device {
        #[cfg(target_os = "macos")]
        Device::Metal => {
            let mut b = crate::metal::Metal::new()?;
            model_suite(spec, device, &mut b, opts, batches, seq_len, r)?;
            for (k, n, t) in b.take_profile() {
                eprintln!("  profile {k:<14} {n:>7} calls {:>10.2} ms", t * 1e3);
            }
            Ok(())
        }
        #[cfg(not(target_os = "macos"))]
        Device::Metal => Err("Metal is only available on macOS".into()),
        Device::Cpu => model_suite(
            spec,
            device,
            &mut crate::cpu::Cpu::new(),
            opts,
            batches,
            seq_len,
            r,
        ),
    }
}

// ------------------------------------------------------------------ engine: scheduling

/// One query through the engine (retrying 429s); returns (total, queue) in ms.
pub fn engine_query(e: &Engine, ids: &[u32]) -> (f64, f64) {
    let t = Instant::now();
    loop {
        match e.submit(vec![ids.to_vec()], Kind::Query) {
            Ok(tk) => {
                let out = tk.wait().expect("query failed");
                return (t.elapsed().as_secs_f64() * 1e3, out.queue_us as f64 / 1e3);
            }
            Err(SubmitError::Busy { .. }) => std::thread::yield_now(),
            Err(e) => panic!("{e:?}"),
        }
    }
}

pub fn start_engine(spec: &Spec, cfg: EngineConfig) -> Result<Arc<Engine>, String> {
    let e = Engine::start(spec.clone(), cfg)?;
    e.wait_ready()?;
    Ok(e)
}

/// Query latency after idle gaps.
pub fn idle_suite(
    spec: &Spec,
    cfg: EngineConfig,
    gaps: &[u64],
    opts: Opts,
    r: &mut Recorder,
) -> Result<(), String> {
    r.scope(spec, cfg.device, "idle");
    let e = start_engine(spec, cfg)?;
    let q = Inputs::new(spec, 1)?.query;
    for &g in gaps {
        let mut p50 = vec![];
        for _ in 0..opts.rounds {
            for _ in 0..3 {
                engine_query(&e, &q);
            }
            let s: Vec<f64> = (0..opts.iters)
                .map(|_| {
                    if g > 0 {
                        std::thread::sleep(Duration::from_millis(g));
                    }
                    engine_query(&e, &q).0
                })
                .collect();
            p50.push(pct(&s, 0.5));
        }
        r.add(&format!("query_after_{g}ms_idle_p50"), "ms", &p50, true);
    }
    Ok(())
}

#[derive(Clone, Copy, Debug)]
pub struct Load {
    pub clients: usize,
    pub doc_batch: usize,
    pub seq_len: usize,
    pub query_every: Duration,
}

impl Default for Load {
    fn default() -> Self {
        Load {
            clients: 4,
            doc_batch: 32,
            seq_len: 128,
            query_every: Duration::from_millis(10),
        }
    }
}

/// Query latency while document clients keep the engine saturated.
pub fn load_suite(
    spec: &Spec,
    cfg: EngineConfig,
    load: Load,
    opts: Opts,
    r: &mut Recorder,
) -> Result<(), String> {
    r.scope(
        spec,
        cfg.device,
        &format!("load_slice{}ms", cfg.doc_slice_ms),
    );
    let e = start_engine(spec, cfg)?;
    let inp = Inputs::new(spec, load.seq_len)?;
    let queries = |dur: f64| {
        let (mut lat, mut queue) = (vec![], vec![]);
        let end = Instant::now() + Duration::from_secs_f64(dur);
        let mut next = Instant::now();
        while Instant::now() < end {
            let (t, q) = engine_query(&e, &inp.query);
            lat.push(t);
            queue.push(q);
            next += load.query_every;
            match next.checked_duration_since(Instant::now()) {
                Some(d) => std::thread::sleep(d),
                None => next = Instant::now(),
            }
        }
        (lat, queue)
    };
    for _ in 0..5 {
        engine_query(&e, &inp.query);
    }
    let rounds = opts.rounds.min(3);
    let mut idle_p99 = vec![];
    for _ in 0..rounds {
        idle_p99.push(pct(&queries(opts.seconds / 2.0).0, 0.99));
    }

    let stop = Arc::new(AtomicBool::new(false));
    let done = Arc::new(AtomicU64::new(0));
    let rejected = Arc::new(AtomicU64::new(0));
    let hs: Vec<_> = (0..load.clients)
        .map(|_| {
            let (e, stop, done, rej) = (e.clone(), stop.clone(), done.clone(), rejected.clone());
            let seqs: Vec<Vec<u32>> = (0..load.doc_batch).map(|_| inp.doc.clone()).collect();
            let toks = (load.doc_batch * inp.doc.len()) as u64;
            std::thread::spawn(move || {
                while !stop.load(Relaxed) {
                    match e.submit(seqs.clone(), Kind::Document) {
                        Ok(t) => {
                            t.wait().unwrap();
                            done.fetch_add(toks, Relaxed);
                        }
                        Err(SubmitError::Busy { .. }) => {
                            rej.fetch_add(1, Relaxed);
                            std::thread::sleep(Duration::from_millis(2));
                        }
                        Err(e) => panic!("{e:?}"),
                    }
                }
            })
        })
        .collect();
    std::thread::sleep(Duration::from_millis(500)); // let the backlog build
    let (mut p50, mut p99, mut q99, mut tps) = (vec![], vec![], vec![], vec![]);
    for _ in 0..rounds {
        let (t0, tok0) = (Instant::now(), done.load(Relaxed));
        let (lat, queue) = queries(opts.seconds);
        tps.push((done.load(Relaxed) - tok0) as f64 / t0.elapsed().as_secs_f64());
        p50.push(pct(&lat, 0.5));
        p99.push(pct(&lat, 0.99));
        q99.push(pct(&queue, 0.99));
    }
    stop.store(true, Relaxed);
    for h in hs {
        h.join().map_err(|_| "document client panicked")?;
    }
    r.add("idle_query_p99", "ms", &idle_p99, true);
    r.add("loaded_query_p50", "ms", &p50, true);
    r.add("loaded_query_p99", "ms", &p99, true);
    r.add("loaded_queue_p99", "ms", &q99, true);
    r.add("ingest_while_loaded", "tok/s", &tps, false);
    r.add(
        "rejected_429s",
        "count",
        &[rejected.load(Relaxed) as f64],
        true,
    );
    Ok(())
}

// ------------------------------------------------------------------ http: transport + encoding

/// A minimal keep-alive HTTP/1.1 client.
struct Client<S: Read + Write> {
    r: BufReader<S>,
}

impl<S: Read + Write> Client<S> {
    /// Returns the status and the server-side time (`X-Inbed-Queue-Us` + `X-Inbed-Compute-Us`).
    fn request(&mut self, req: &[u8]) -> Result<(u16, f64), String> {
        self.r.get_mut().write_all(req).map_err(|e| e.to_string())?;
        let mut line = String::new();
        self.r.read_line(&mut line).map_err(|e| e.to_string())?;
        let status = line
            .split_whitespace()
            .nth(1)
            .and_then(|s| s.parse().ok())
            .ok_or("bad status line")?;
        let (mut len, mut server_us) = (0, 0.0);
        loop {
            line.clear();
            self.r.read_line(&mut line).map_err(|e| e.to_string())?;
            let h = line.trim_end().to_ascii_lowercase();
            if h.is_empty() {
                break;
            }
            let Some((k, v)) = h.split_once(':') else {
                continue;
            };
            match k {
                "content-length" => len = v.trim().parse().map_err(|_| "bad content-length")?,
                "x-inbed-queue-us" | "x-inbed-compute-us" => {
                    server_us += v.trim().parse::<f64>().unwrap_or(0.0)
                }
                _ => {}
            }
        }
        let mut body = vec![0; len];
        self.r.read_exact(&mut body).map_err(|e| e.to_string())?;
        Ok((status, server_us))
    }
}

fn post(body: &str, binary: bool) -> Vec<u8> {
    let accept = if binary {
        "application/octet-stream, application/json;q=0.5"
    } else {
        "application/json"
    };
    format!(
        "POST /v1/embeddings HTTP/1.1\r\nHost: x\r\nAccept: {accept}\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{body}",
        body.len()
    )
    .into_bytes()
}

/// Round trips through an in-process server, over TCP and a Unix socket, JSON and binary:
/// `/health` (server overhead alone), a 1-text query, and a 128-text document batch. Besides the
/// round trip, `overhead` subtracts the server's reported queue and compute time, leaving what
/// transport, parsing and encoding cost, independent of the model's speed.
pub fn http_suite(
    spec: &Spec,
    cfg: EngineConfig,
    opts: Opts,
    r: &mut Recorder,
) -> Result<(), String> {
    r.scope(spec, cfg.device, "http");
    let e = start_engine(spec, cfg)?;
    let api = Arc::new(Api {
        engines: vec![e],
        max_inputs: 2048,
    });
    let handler: crate::http::Handler = Arc::new(move |q| api.handle(q));
    let hcfg = crate::http::Config::default();
    let active = Arc::new(AtomicUsize::new(0));
    let tcp = std::net::TcpListener::bind("127.0.0.1:0").map_err(|e| e.to_string())?;
    let addr = tcp.local_addr().map_err(|e| e.to_string())?;
    crate::http::spawn(tcp, hcfg, active.clone(), handler.clone());
    let sock = std::env::temp_dir().join(format!("inbed-bench-{}.sock", std::process::id()));
    let sock = sock.to_str().ok_or("temp dir is not UTF-8")?.to_string();
    crate::http::spawn(crate::http::bind_unix(&sock, 0o600)?, hcfg, active, handler);

    let query = request_body(&[QUERY_TEXT], true);
    let doc_text = DOC_TEXT.repeat(2);
    let docs: Vec<&str> = (0..128).map(|_| doc_text.as_str()).collect();
    let batch = request_body(&docs, false);
    let health = b"GET /health HTTP/1.1\r\nHost: x\r\n\r\n".to_vec();
    // (name, request, samples per round, rounds): big batches are compute-bound, so few of them
    let few = (opts.iters / 10).max(3);
    let cases: Vec<(&str, Vec<u8>, usize, usize)> = vec![
        ("health", health, opts.iters * 10, opts.rounds),
        (
            "query_json",
            post(&query, false),
            opts.iters * 2,
            opts.rounds,
        ),
        (
            "query_binary",
            post(&query, true),
            opts.iters * 2,
            opts.rounds,
        ),
        (
            "batch128_json",
            post(&batch, false),
            few,
            opts.rounds.min(3),
        ),
        (
            "batch128_binary",
            post(&batch, true),
            few,
            opts.rounds.min(3),
        ),
    ];
    struct Rounds {
        p50: Vec<f64>,
        p99: Vec<f64>,
        overhead: Vec<f64>,
    }
    fn run<S: Read + Write>(
        mut c: Client<S>,
        req: &[u8],
        n: usize,
        rounds: usize,
    ) -> Result<Rounds, String> {
        c.request(req)?;
        let mut out = Rounds {
            p50: vec![],
            p99: vec![],
            overhead: vec![],
        };
        for _ in 0..rounds {
            let (mut rtt, mut over) = (Vec::with_capacity(n), Vec::with_capacity(n));
            for _ in 0..n {
                let t = Instant::now();
                let (status, server_us) = c.request(req)?;
                if status != 200 {
                    return Err(format!("HTTP {status}"));
                }
                let us = t.elapsed().as_secs_f64() * 1e6;
                rtt.push(us);
                over.push(us - server_us);
            }
            out.p50.push(pct(&rtt, 0.5));
            out.p99.push(pct(&rtt, 0.99));
            out.overhead.push(pct(&over, 0.5));
        }
        Ok(out)
    }
    for (name, req, n, rounds) in &cases {
        for transport in ["tcp", "unix"] {
            let res = if transport == "tcp" {
                let s = std::net::TcpStream::connect(addr).map_err(|e| e.to_string())?;
                s.set_nodelay(true).ok();
                run(
                    Client {
                        r: BufReader::new(s),
                    },
                    req,
                    *n,
                    *rounds,
                )?
            } else {
                let s =
                    std::os::unix::net::UnixStream::connect(&sock).map_err(|e| e.to_string())?;
                run(
                    Client {
                        r: BufReader::new(s),
                    },
                    req,
                    *n,
                    *rounds,
                )?
            };
            r.add(&format!("{transport}_{name}_p50"), "us", &res.p50, true);
            r.add(&format!("{transport}_{name}_p99"), "us", &res.p99, true);
            if *name != "health" {
                r.add(
                    &format!("{transport}_{name}_overhead_p50"),
                    "us",
                    &res.overhead,
                    true,
                );
            }
        }
    }
    let _ = std::fs::remove_file(&sock);
    Ok(())
}

fn request_body(texts: &[&str], query: bool) -> String {
    let mut s = String::from("{\"input\":[");
    for (i, t) in texts.iter().enumerate() {
        if i > 0 {
            s.push(',');
        }
        crate::json::write_str(&mut s, t);
    }
    s.push_str("],\"input_type\":");
    s.push_str(if query { "\"query\"}" } else { "\"document\"}" });
    s
}

// ------------------------------------------------------------------ run metadata, files, compare

fn cmd(prog: &str, args: &[&str]) -> Option<String> {
    let o = std::process::Command::new(prog).args(args).output().ok()?;
    o.status
        .success()
        .then(|| String::from_utf8_lossy(&o.stdout).trim().to_string())
}

/// 1-minute load average.
pub fn load_avg() -> Option<f64> {
    let s = if cfg!(target_os = "macos") {
        cmd("sysctl", &["-n", "vm.loadavg"])?
    } else {
        std::fs::read_to_string("/proc/loadavg").ok()?
    };
    s.trim_matches(|c| c == '{' || c == ' ')
        .split_whitespace()
        .next()?
        .parse()
        .ok()
}

/// The busiest other processes, `(cpu %, name)`, to explain a noisy run.
pub fn busy_processes(min_cpu: f64) -> Vec<(f64, String)> {
    let out = cmd("ps", &["-Ao", "pcpu=,comm="]).unwrap_or_default();
    let me = std::process::id();
    let mine = cmd("ps", &["-o", "comm=", "-p", &me.to_string()]).unwrap_or_default();
    let mut v: Vec<(f64, String)> = out
        .lines()
        .filter_map(|l| {
            let l = l.trim();
            let (c, n) = l.split_once(' ')?;
            Some((c.parse().ok()?, n.trim().to_string()))
        })
        .filter(|(c, n)| *c >= min_cpu && *n != mine)
        .collect();
    v.sort_by(|a, b| b.0.total_cmp(&a.0));
    v.truncate(5);
    v
}

fn utc_now() -> String {
    let secs = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0) as i64;
    let (days, rem) = (secs.div_euclid(86400), secs.rem_euclid(86400));
    // civil-from-days (Howard Hinnant)
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = yoe + era * 400 + (m <= 2) as i64;
    format!(
        "{y:04}-{m:02}-{d:02}T{:02}:{:02}:{:02}Z",
        rem / 3600,
        rem / 60 % 60,
        rem % 60
    )
}

/// Where and how a run happened: enough to tell whether two runs are comparable.
pub fn meta(models: &[&Spec], device: Device, opts: Opts, args: &[String]) -> Json {
    let cpu = if cfg!(target_os = "macos") {
        cmd("sysctl", &["-n", "machdep.cpu.brand_string"])
    } else {
        std::fs::read_to_string("/proc/cpuinfo").ok().and_then(|s| {
            s.lines()
                .find(|l| l.starts_with("model name"))
                .and_then(|l| l.split(':').nth(1))
                .map(|s| s.trim().to_string())
        })
    };
    #[cfg(target_os = "macos")]
    let gpu = (device == Device::Metal)
        .then(|| {
            crate::metal::Metal::new()
                .ok()
                .map(|m| m.device_name().to_string())
        })
        .flatten();
    #[cfg(not(target_os = "macos"))]
    let gpu: Option<String> = None;
    let env: Vec<Json> = std::env::vars()
        .filter(|(k, _)| k.starts_with("INBED_"))
        .map(|(k, v)| Json::from(format!("{k}={v}")))
        .collect();
    let models: Vec<Json> = models
        .iter()
        .map(|s| {
            crate::obj! {
                "id" => s.id.as_str(),
                // the snapshot directory name is the Hugging Face revision
                "revision" => s.dir.file_name().and_then(|f| f.to_str()).unwrap_or(""),
            }
        })
        .collect();
    let opt = |s: Option<String>| s.map(Json::from).unwrap_or(Json::Null);
    crate::obj! {
        "date" => utc_now(),
        "git" => opt(cmd("git", &["rev-parse", "--short=12", "HEAD"])),
        "git_dirty" => cmd("git", &["status", "--porcelain", "--untracked-files=no"]).map(|s| !s.is_empty()).unwrap_or(false),
        "os" => format!("{} {}", std::env::consts::OS, cmd("uname", &["-r"]).unwrap_or_default()),
        "arch" => std::env::consts::ARCH,
        "cpu" => opt(cpu),
        "gpu" => opt(gpu),
        "device" => format!("{device:?}").to_lowercase(),
        "load_avg_1m" => load_avg().map(Json::Num).unwrap_or(Json::Null),
        "rounds" => opts.rounds,
        "iters" => opts.iters,
        "quick" => opts.quick,
        "models" => Json::Arr(models),
        "env" => Json::Arr(env),
        "args" => Json::Arr(args.iter().map(|a| Json::from(a.as_str())).collect()),
    }
}

pub fn to_json(meta: Json, metrics: &[Metric]) -> Json {
    let ms: Vec<Json> = metrics
        .iter()
        .map(|m| {
            crate::obj! {
                "name" => m.name.as_str(),
                "unit" => m.unit,
                "value" => m.value,
                "lo" => m.lo,
                "hi" => m.hi,
                "lower_is_better" => m.lower_is_better,
            }
        })
        .collect();
    crate::obj! {"meta" => meta, "metrics" => Json::Arr(ms)}
}

pub fn metrics_from_json(j: &Json) -> Result<Vec<Metric>, String> {
    let arr = j
        .get("metrics")
        .and_then(Json::as_arr)
        .ok_or("no \"metrics\" array")?;
    arr.iter()
        .map(|m| {
            let f = |k: &str| {
                m.get(k)
                    .and_then(Json::as_f64)
                    .ok_or(format!("metric missing {k}"))
            };
            Ok(Metric {
                name: m
                    .get("name")
                    .and_then(Json::as_str)
                    .ok_or("metric missing name")?
                    .to_string(),
                unit: "",
                value: f("value")?,
                lo: f("lo")?,
                hi: f("hi")?,
                lower_is_better: m
                    .get("lower_is_better")
                    .and_then(Json::as_bool)
                    .unwrap_or(true),
            })
        })
        .collect()
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Verdict {
    Better,
    Worse,
    Same,
}

/// A change counts only if it is larger than `threshold` (relative) and the two runs' spreads
/// don't overlap.
pub fn verdict(base: &Metric, now: &Metric, threshold: f64) -> Verdict {
    if base.value == 0.0 && now.value == 0.0 {
        return Verdict::Same;
    }
    // from zero, any change is a full-size change
    let rel = if base.value == 0.0 {
        now.value.signum()
    } else {
        (now.value - base.value) / base.value.abs()
    };
    let overlap = now.lo <= base.hi && base.lo <= now.hi;
    if rel.abs() < threshold || overlap {
        return Verdict::Same;
    }
    let up = rel > 0.0;
    if up != base.lower_is_better {
        Verdict::Better
    } else {
        Verdict::Worse
    }
}

/// Print a comparison table; returns the number of regressions.
pub fn compare(base: &[Metric], now: &[Metric], threshold: f64) -> usize {
    let mut worse = 0;
    println!(
        "{:<58} {:>11} {:>11} {:>8}",
        "metric", "baseline", "now", "change"
    );
    for m in now {
        let Some(b) = base.iter().find(|b| b.name == m.name) else {
            println!(
                "{:<58} {:>11} {:>11} {:>8}",
                m.name,
                "-",
                num(m.value),
                "new"
            );
            continue;
        };
        let v = verdict(b, m, threshold);
        if v == Verdict::Worse {
            worse += 1;
        }
        let change = if b.value == 0.0 {
            if m.value == 0.0 {
                "0.0%".into()
            } else {
                "from 0".into()
            }
        } else {
            format!("{:+.1}%", (m.value - b.value) / b.value.abs() * 100.0)
        };
        println!(
            "{:<58} {:>11} {:>11} {:>8} {}",
            m.name,
            num(b.value),
            num(m.value),
            change,
            match v {
                Verdict::Better => "better",
                Verdict::Worse => "WORSE",
                Verdict::Same => "",
            }
        );
    }
    worse
}

#[cfg(test)]
mod tests {
    use super::*;

    fn m(v: f64, lo: f64, hi: f64) -> Metric {
        Metric {
            name: "x".into(),
            unit: "ms",
            value: v,
            lo,
            hi,
            lower_is_better: true,
        }
    }

    #[test]
    fn verdicts() {
        let base = m(10.0, 9.8, 10.2);
        assert_eq!(verdict(&base, &m(10.1, 9.9, 10.3), 0.05), Verdict::Same);
        assert_eq!(verdict(&base, &m(12.0, 11.8, 12.3), 0.05), Verdict::Worse);
        assert_eq!(verdict(&base, &m(8.0, 7.9, 8.1), 0.05), Verdict::Better);
        // large change but the spreads overlap: noise, not a verdict
        assert_eq!(verdict(&base, &m(12.0, 9.0, 15.0), 0.05), Verdict::Same);
        assert_eq!(
            verdict(&m(0.0, 0.0, 0.0), &m(0.0, 0.0, 0.0), 0.05),
            Verdict::Same
        );
        assert_eq!(
            verdict(&m(0.0, 0.0, 0.0), &m(3.0, 3.0, 3.0), 0.05),
            Verdict::Worse
        );
    }

    #[test]
    fn json_roundtrip_and_date() {
        let ms = vec![m(1.5, 1.0, 2.0)];
        let j = to_json(Json::Null, &ms);
        let back = metrics_from_json(&Json::parse(&j.to_string()).unwrap()).unwrap();
        assert_eq!(back[0].value, 1.5);
        assert_eq!(back[0].hi, 2.0);
        let d = utc_now();
        assert_eq!(d.len(), 20);
        assert!(d.starts_with("20") && d.ends_with('Z'));
    }
}
