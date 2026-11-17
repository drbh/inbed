//! One worker thread per model owns the backend and runs packed batches.
//!
//! The request path is lock-free: callers push jobs into one of two bounded rings
//! (queries, documents), admission is by expected wait, and replies come back through one-shot
//! slots. The worker runs queries first and document work in time-sized slices, so a query waits
//! behind at most one slice.

use crate::backend::{Backend, Batch};
use crate::model::{self, Spec, Weights};
use crate::ring::{Promise, Ring, Ticket, oneshot};
use crate::tokenizer::Tokenizer;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering, fence};
use std::sync::{Arc, OnceLock};
use std::thread::Thread;
use std::time::{Duration, Instant};

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Device {
    Metal,
    Cpu,
}

impl Device {
    pub fn parse(s: &str) -> Result<Device, String> {
        match s {
            "metal" | "gpu" => Ok(Device::Metal),
            "cpu" => Ok(Device::Cpu),
            "auto" => Ok(Device::auto()),
            _ => Err(format!("unknown device {s:?} (metal, cpu, auto)")),
        }
    }
    pub fn auto() -> Device {
        if cfg!(target_os = "macos") {
            Device::Metal
        } else {
            Device::Cpu
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct EngineConfig {
    pub device: Device,
    /// tokens per forward pass for queries, and the cap on a document slice
    /// (a single longer sequence still runs, alone)
    pub max_batch_tokens: usize,
    /// ring capacity per kind (jobs); a full ring is a 429
    pub queue: usize,
    /// admit a query only if its expected wait is under this
    pub query_max_wait_ms: u64,
    /// admit a document request only if its expected wait is under this
    pub doc_max_wait_ms: u64,
    /// target GPU time of one document slice; queries run between slices
    pub doc_slice_ms: u64,
    /// how long the idle worker spins before parking. Longer keeps the CPU core clocked up, which
    /// matters on Apple silicon: after ~50 ms idle, Qwen3's command encoding slows from 2 to 9 ms
    pub spin_us: u64,
}

impl Default for EngineConfig {
    fn default() -> Self {
        EngineConfig {
            device: Device::auto(),
            max_batch_tokens: 16384,
            queue: 64,
            query_max_wait_ms: 100,
            doc_max_wait_ms: 5000,
            doc_slice_ms: 20,
            spin_us: 50,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    Query = 0,
    Document = 1,
}

pub struct Embedded {
    /// one full-dims, normalized vector per sequence
    pub vecs: Vec<Vec<f32>>,
    /// from submit to the first forward pass that included this request
    pub queue_us: u64,
    /// sum of the forward passes that included this request
    pub compute_us: u64,
}

pub type Reply = Result<Embedded, String>;

fn stopped() -> Reply {
    Err("model worker stopped".into())
}

struct Job {
    seqs: Vec<Vec<u32>>,
    enqueued: Instant,
    reply: Promise<Reply>,
}

#[derive(Default)]
pub struct Stats {
    pub requests: AtomicU64,
    pub sequences: AtomicU64,
    pub tokens: AtomicU64,
    pub batches: AtomicU64,
    pub busy_us: AtomicU64,
    /// 429s from admission control or a full ring
    pub rejected: AtomicU64,
}

/// Everything the HTTP threads and the worker share. All of it is atomics or set-once.
struct Shared {
    rings: [Ring<Job>; 2],
    /// tokens admitted and not yet computed, per kind
    queued: [AtomicU64; 2],
    /// measured throughput (f64 bits, tokens/s) and small-batch latency (µs), both EWMAs
    tps: AtomicU64,
    base_us: AtomicU64,
    sleeping: AtomicBool,
    shutdown: AtomicBool,
    worker: OnceLock<Thread>,
    ready: AtomicBool,
    failed: OnceLock<String>,
    stats: Stats,
}

impl Shared {
    fn tps(&self) -> f64 {
        f64::from_bits(self.tps.load(Ordering::Relaxed)).max(1.0)
    }
    fn estimate_ms(&self, tokens: u64) -> u64 {
        let base = self.base_us.load(Ordering::Relaxed) as f64 / 1e3;
        (base + tokens as f64 / self.tps() * 1e3) as u64
    }
    fn wake(&self) {
        // pairs with the fence in `idle`: either we see `sleeping`, or the worker sees our push
        fence(Ordering::SeqCst);
        if self.sleeping.load(Ordering::Relaxed)
            && let Some(t) = self.worker.get()
        {
            t.unpark();
        }
    }
}

pub struct Engine {
    pub spec: Spec,
    pub tokenizer: Tokenizer,
    pub device: Device,
    pub cfg: EngineConfig,
    sh: Arc<Shared>,
}

#[derive(Debug)]
pub enum SubmitError {
    /// over the wait limit or the ring is full; retry after roughly this long
    Busy {
        retry_after_ms: u64,
    },
    NotReady(String),
    Failed(String),
}

impl Engine {
    /// Load the tokenizer now; start the worker, which loads weights on its own thread.
    pub fn start(spec: Spec, cfg: EngineConfig) -> Result<Arc<Engine>, String> {
        let tokenizer = spec.tokenizer()?;
        let sh = Arc::new(Shared {
            rings: [Ring::new(cfg.queue), Ring::new(cfg.queue)],
            queued: [AtomicU64::new(0), AtomicU64::new(0)],
            tps: AtomicU64::new(1000f64.to_bits()),
            base_us: AtomicU64::new(0),
            sleeping: AtomicBool::new(false),
            shutdown: AtomicBool::new(false),
            worker: OnceLock::new(),
            ready: AtomicBool::new(false),
            failed: OnceLock::new(),
            stats: Stats::default(),
        });
        let e = Arc::new(Engine {
            spec: spec.clone(),
            tokenizer,
            device: cfg.device,
            cfg,
            sh: sh.clone(),
        });
        std::thread::Builder::new()
            .name(format!("inbed-{}", spec.id))
            .spawn(move || {
                crate::sys::latency_critical_thread();
                let _ = sh.worker.set(std::thread::current());
                let r = match cfg.device {
                    #[cfg(target_os = "macos")]
                    Device::Metal => {
                        crate::metal::Metal::new().and_then(|b| worker(b, &spec, cfg, &sh))
                    }
                    #[cfg(not(target_os = "macos"))]
                    Device::Metal => Err("Metal is only available on macOS".to_string()),
                    Device::Cpu => worker(crate::cpu::Cpu::new(), &spec, cfg, &sh),
                };
                if let Err(e) = r {
                    eprintln!("inbed: model {} failed: {e}", spec.id);
                    let _ = sh.failed.set(e);
                }
                // fail anything still queued (their promises reply on drop)
                for r in &sh.rings {
                    while r.pop().is_some() {}
                }
            })
            .map_err(|e| e.to_string())?;
        Ok(e)
    }

    pub fn stats(&self) -> &Stats {
        &self.sh.stats
    }

    pub fn is_ready(&self) -> bool {
        self.sh.ready.load(Ordering::Acquire)
    }

    pub fn failure(&self) -> Option<String> {
        self.sh.failed.get().cloned()
    }

    /// Tokens admitted and not yet computed.
    pub fn queued_tokens(&self, kind: Kind) -> u64 {
        self.sh.queued[kind as usize].load(Ordering::Relaxed)
    }

    /// Measured throughput, tokens/s.
    pub fn throughput(&self) -> f64 {
        self.sh.tps()
    }

    /// Block until the weights are loaded (or loading failed). Startup only.
    pub fn wait_ready(&self) -> Result<(), String> {
        while !self.is_ready() {
            if let Some(e) = self.failure() {
                return Err(e);
            }
            std::thread::sleep(Duration::from_millis(20));
        }
        Ok(())
    }

    /// Token ids for one input, with the prompt prefix for its input type, truncated to max_tokens.
    pub fn tokenize(&self, text: &str, query: bool) -> Vec<u32> {
        let prefix = if query {
            &self.spec.query_prefix
        } else {
            &self.spec.document_prefix
        };
        if prefix.is_empty() {
            self.tokenizer.encode(text, self.spec.max_tokens)
        } else {
            self.tokenizer
                .encode(&format!("{prefix}{text}"), self.spec.max_tokens)
        }
    }

    /// Admit and queue a request. The ticket must be waited on by the calling thread.
    pub fn submit(&self, seqs: Vec<Vec<u32>>, kind: Kind) -> Result<Ticket<Reply>, SubmitError> {
        let sh = &*self.sh;
        if let Some(e) = sh.failed.get() {
            return Err(SubmitError::Failed(e.clone()));
        }
        if !self.is_ready() {
            return Err(SubmitError::NotReady(format!(
                "model {} is still loading",
                self.spec.id
            )));
        }
        let k = kind as usize;
        let tokens: u64 = seqs.iter().map(|s| s.len() as u64).sum();
        let before = sh.queued[k].fetch_add(tokens, Ordering::AcqRel);
        // queries wait behind queries and at most one document slice; documents behind everything
        let (ahead, extra_ms, limit) = match kind {
            Kind::Query => {
                let slice = if sh.queued[1].load(Ordering::Relaxed) > 0 {
                    self.cfg.doc_slice_ms
                } else {
                    0
                };
                (before + tokens, slice, self.cfg.query_max_wait_ms)
            }
            Kind::Document => (
                before + tokens + sh.queued[0].load(Ordering::Relaxed),
                0,
                self.cfg.doc_max_wait_ms,
            ),
        };
        let est = sh.estimate_ms(ahead) + extra_ms;
        // an empty queue always admits, so a request larger than the limit can still run
        if before > 0 && est > limit {
            sh.queued[k].fetch_sub(tokens, Ordering::AcqRel);
            sh.stats.rejected.fetch_add(1, Ordering::Relaxed);
            return Err(SubmitError::Busy {
                retry_after_ms: est - limit,
            });
        }
        let (reply, ticket) = oneshot(stopped);
        let job = Job {
            seqs,
            enqueued: Instant::now(),
            reply,
        };
        if let Err(job) = sh.rings[k].push(job) {
            sh.queued[k].fetch_sub(tokens, Ordering::AcqRel);
            sh.stats.rejected.fetch_add(1, Ordering::Relaxed);
            drop((job, ticket));
            return Err(SubmitError::Busy {
                retry_after_ms: est.max(1),
            });
        }
        sh.wake();
        Ok(ticket)
    }

    /// Submit as documents and wait (CLI and tests).
    pub fn embed(&self, seqs: Vec<Vec<u32>>) -> Result<Vec<Vec<f32>>, String> {
        match self.submit(seqs, Kind::Document) {
            Ok(t) => t.wait().map(|e| e.vecs),
            Err(SubmitError::Busy { .. }) => Err("queue full".into()),
            Err(SubmitError::NotReady(e)) | Err(SubmitError::Failed(e)) => Err(e),
        }
    }
}

impl Drop for Engine {
    fn drop(&mut self) {
        self.sh.shutdown.store(true, Ordering::Release);
        if let Some(t) = self.sh.worker.get() {
            t.unpark();
        }
    }
}

/// Runs forward passes and keeps the throughput estimates current.
struct Exec<'a, B: Backend> {
    b: B,
    w: Weights<B::W>,
    spec: &'a Spec,
    sh: &'a Shared,
    long_seq: usize,
}

impl<B: Backend> Exec<'_, B> {
    /// One forward pass; `Err` if it panicked. Returns the vectors and its wall time in µs.
    fn forward(&mut self, seqs: &[&[u32]]) -> Result<(Vec<Vec<f32>>, u64), String> {
        let t = Instant::now();
        let (spec, w, b) = (self.spec, &self.w, &mut self.b);
        let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            model::forward(spec, w, b, &Batch::new(seqs))
        }));
        let us = t.elapsed().as_micros() as u64;
        let tokens: usize = seqs.iter().map(|s| s.len()).sum();
        let st = &self.sh.stats;
        st.batches.fetch_add(1, Ordering::Relaxed);
        st.busy_us.fetch_add(us, Ordering::Relaxed);
        st.tokens.fetch_add(tokens as u64, Ordering::Relaxed);
        st.sequences.fetch_add(seqs.len() as u64, Ordering::Relaxed);
        self.observe(tokens, us);
        match r {
            Ok(v) => Ok((v, us)),
            Err(p) => {
                let msg = p
                    .downcast_ref::<String>()
                    .cloned()
                    .or_else(|| p.downcast_ref::<&str>().map(|s| s.to_string()))
                    .unwrap_or_else(|| "forward pass panicked".into());
                eprintln!("inbed: forward failed: {msg}");
                Err(msg)
            }
        }
    }

    /// Fold a measurement into the estimates: small batches give latency, big ones throughput.
    fn observe(&self, tokens: usize, us: u64) {
        let ewma = |a: &AtomicU64, x: f64| {
            let old = f64::from_bits(a.load(Ordering::Relaxed));
            a.store((old * 0.8 + x * 0.2).to_bits(), Ordering::Relaxed);
        };
        if tokens >= 256 && us > 0 {
            ewma(&self.sh.tps, tokens as f64 / (us as f64 / 1e6));
        }
        if tokens <= 64 {
            let old = self.sh.base_us.load(Ordering::Relaxed) as f64;
            self.sh
                .base_us
                .store((old * 0.8 + us as f64 * 0.2) as u64, Ordering::Relaxed);
        }
    }

    /// Seed the estimates before taking traffic.
    fn calibrate(&mut self) {
        let short: Vec<u32> = (100..108).collect();
        let n = 128.min(self.spec.max_tokens);
        let doc: Vec<u32> = (0..n as u32).map(|i| 100 + i % 1000).collect();
        let _ = self.forward(&[&short]); // warm-up: first dispatches pay pipeline / allocation costs
        let base = (0..3)
            .filter_map(|_| self.forward(&[&short]).ok().map(|r| r.1))
            .min()
            .unwrap_or(0);
        self.sh.base_us.store(base, Ordering::Relaxed);
        let seqs: Vec<&[u32]> = vec![&doc; 256 / n.max(1) + 1];
        let tokens: usize = seqs.iter().map(|s| s.len()).sum();
        if let Ok((_, us)) = self.forward(&seqs) {
            let tps = tokens as f64 / (us.max(1) as f64 / 1e6);
            self.sh.tps.store(tps.to_bits(), Ordering::Relaxed);
        }
    }
}

/// Per-request bookkeeping while the worker runs it, possibly over several forward passes.
struct Work {
    job: Job,
    kind: Kind,
    next: usize,
    out: Vec<Vec<f32>>,
    started: Option<Instant>,
    compute_us: u64,
    error: Option<String>,
}

impl Work {
    fn new(job: Job, kind: Kind) -> Work {
        let n = job.seqs.len();
        Work {
            job,
            kind,
            next: 0,
            out: vec![Vec::new(); n],
            started: None,
            compute_us: 0,
            error: None,
        }
    }
    fn remaining_tokens(&self) -> usize {
        self.job.seqs[self.next..].iter().map(Vec::len).sum()
    }
    fn finished(&self) -> bool {
        self.next == self.job.seqs.len()
    }
    fn complete(self) {
        let r = match self.error {
            Some(e) => Err(e),
            None => Ok(Embedded {
                vecs: self.out,
                queue_us: self
                    .started
                    .map(|s| s.duration_since(self.job.enqueued).as_micros() as u64)
                    .unwrap_or(0),
                compute_us: self.compute_us,
            }),
        };
        self.job.reply.send(r);
    }
}

/// Run `works` from their `next` sequence: pack at most `budget` tokens per forward pass (a single
/// longer sequence runs alone), never mixing long and short sequences, and stop after `max_passes`.
fn run<B: Backend>(ex: &mut Exec<B>, works: &mut [Work], budget: usize, max_passes: usize) {
    for _ in 0..max_passes {
        // contiguous picks across works, in order
        let mut pick: Vec<(usize, usize)> = Vec::new();
        let mut t = 0;
        let mut long = None;
        'outer: for (wi, w) in works.iter().enumerate() {
            for si in w.next..w.job.seqs.len() {
                let len = w.job.seqs[si].len();
                let is_long = len >= ex.long_seq;
                if !pick.is_empty() && (t + len > budget || long != Some(is_long)) {
                    break 'outer;
                }
                long = Some(is_long);
                pick.push((wi, si));
                t += len;
            }
        }
        if pick.is_empty() {
            return;
        }
        let started = Instant::now();
        let seqs: Vec<&[u32]> = pick
            .iter()
            .map(|&(wi, si)| works[wi].job.seqs[si].as_slice())
            .collect();
        let r = ex.forward(&seqs);
        let mut touched: Vec<usize> = pick.iter().map(|p| p.0).collect();
        touched.dedup();
        match r {
            Ok((out, us)) => {
                for (&(wi, si), v) in pick.iter().zip(out) {
                    works[wi].out[si] = v;
                    works[wi].next = si + 1;
                }
                for &wi in &touched {
                    let w = &mut works[wi];
                    w.started.get_or_insert(started);
                    w.compute_us += us;
                }
                for &(wi, si) in &pick {
                    let k = works[wi].kind as usize;
                    ex.sh.queued[k]
                        .fetch_sub(works[wi].job.seqs[si].len() as u64, Ordering::AcqRel);
                }
            }
            Err(e) => {
                // the whole request fails; release everything it still had queued
                for &wi in &touched {
                    let w = &mut works[wi];
                    let rest = w.remaining_tokens() as u64;
                    ex.sh.queued[w.kind as usize].fetch_sub(rest, Ordering::AcqRel);
                    w.next = w.job.seqs.len();
                    w.error = Some(e.clone());
                }
            }
        }
    }
}

fn complete_finished(works: &mut Vec<Work>) {
    let mut i = 0;
    while i < works.len() {
        if works[i].finished() {
            works.remove(i).complete();
        } else {
            i += 1;
        }
    }
}

/// After this many query rounds in a row, one document slice runs if any is waiting.
const QUERY_STREAK: usize = 8;

fn worker<B: Backend>(b: B, spec: &Spec, cfg: EngineConfig, sh: &Shared) -> Result<(), String> {
    let t0 = Instant::now();
    let mut b = b;
    let w = model::load_weights(spec, &mut b)?;
    let mut ex = Exec {
        b,
        w,
        spec,
        sh,
        long_seq: crate::backend::long_seq(),
    };
    ex.calibrate();
    eprintln!(
        "inbed: {} ready on {} ({:.1}s; {:.0} tokens/s, {:.2} ms small batch)",
        spec.id,
        ex.b.name(),
        t0.elapsed().as_secs_f64(),
        sh.tps(),
        sh.base_us.load(Ordering::Relaxed) as f64 / 1e3
    );
    sh.ready.store(true, Ordering::Release);

    let mut docs: Vec<Work> = Vec::new();
    // query rounds since the last document slice: a steady query stream can't starve documents
    let mut streak = 0;
    loop {
        // 1. queries first: everything waiting, up to the token budget
        let docs_waiting = !docs.is_empty() || !sh.rings[Kind::Document as usize].is_empty();
        if (streak < QUERY_STREAK || !docs_waiting)
            && let Some(first) = sh.rings[Kind::Query as usize].pop()
        {
            streak += 1;
            let mut toks: usize = first.seqs.iter().map(Vec::len).sum();
            let mut qs = vec![Work::new(first, Kind::Query)];
            while toks < cfg.max_batch_tokens {
                match sh.rings[Kind::Query as usize].pop() {
                    Some(j) => {
                        toks += j.seqs.iter().map(Vec::len).sum::<usize>();
                        qs.push(Work::new(j, Kind::Query));
                    }
                    None => break,
                }
            }
            sh.stats
                .requests
                .fetch_add(qs.len() as u64, Ordering::Relaxed);
            // shortest first so the most requests finish soonest; passes stay length-homogeneous
            qs.sort_by_key(|w| w.job.seqs.iter().map(Vec::len).max().unwrap_or(0));
            run(&mut ex, &mut qs, cfg.max_batch_tokens, usize::MAX);
            complete_finished(&mut qs);
            continue;
        }

        // 2. one document slice, sized by expected GPU time
        let budget = ((sh.tps() * cfg.doc_slice_ms as f64 / 1e3) as usize)
            .clamp(64, cfg.max_batch_tokens.max(64));
        let mut pending: usize = docs.iter().map(Work::remaining_tokens).sum();
        while pending < budget {
            match sh.rings[Kind::Document as usize].pop() {
                Some(j) => {
                    sh.stats.requests.fetch_add(1, Ordering::Relaxed);
                    let w = Work::new(j, Kind::Document);
                    pending += w.remaining_tokens();
                    docs.push(w);
                }
                None => break,
            }
        }
        streak = 0;
        if !docs.is_empty() {
            run(&mut ex, &mut docs, budget, 1);
            complete_finished(&mut docs);
            continue;
        }

        // 3. idle
        if sh.shutdown.load(Ordering::Acquire) {
            return Ok(());
        }
        idle(sh, cfg.spin_us);
    }
}

fn has_work(sh: &Shared) -> bool {
    !sh.rings[0].is_empty() || !sh.rings[1].is_empty() || sh.shutdown.load(Ordering::Acquire)
}

/// Spin briefly (a request arriving now starts without a wake-up), then park until a producer
/// unparks us. Lost wake-ups are impossible: we publish `sleeping` and re-check the rings across
/// a SeqCst fence, and producers do the reverse in `Shared::wake`.
fn idle(sh: &Shared, spin_us: u64) {
    let until = Instant::now() + Duration::from_micros(spin_us);
    while Instant::now() < until {
        if has_work(sh) {
            return;
        }
        std::hint::spin_loop();
    }
    sh.sleeping.store(true, Ordering::Relaxed);
    fence(Ordering::SeqCst);
    while !has_work(sh) {
        std::thread::park();
    }
    sh.sleeping.store(false, Ordering::Relaxed);
}
