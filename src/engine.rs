//! One worker thread per model owns the backend and runs packed batches; callers submit jobs.

use crate::backend::{Backend, Batch};
use crate::model::{self, Spec};
use crate::tokenizer::Tokenizer;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::mpsc::{self, Receiver, SyncSender, TrySendError};
use std::sync::{Arc, Mutex};
use std::time::Instant;

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
    /// tokens per forward pass (a single longer sequence still runs, alone)
    pub max_batch_tokens: usize,
    /// jobs waiting before submit() reports `Busy`
    pub queue: usize,
}

impl Default for EngineConfig {
    fn default() -> Self {
        EngineConfig {
            device: Device::auto(),
            max_batch_tokens: 16384,
            queue: 64,
        }
    }
}

pub type Reply = Result<Vec<Vec<f32>>, String>;

struct Job {
    seqs: Vec<Vec<u32>>,
    reply: mpsc::Sender<Reply>,
}

#[derive(Default)]
pub struct Stats {
    pub requests: AtomicU64,
    pub sequences: AtomicU64,
    pub tokens: AtomicU64,
    pub batches: AtomicU64,
    pub busy_us: AtomicU64,
}

pub struct Engine {
    pub spec: Spec,
    pub tokenizer: Tokenizer,
    tx: SyncSender<Job>,
    ready: Arc<AtomicBool>,
    failed: Arc<Mutex<Option<String>>>,
    pub stats: Arc<Stats>,
    pub device: Device,
}

pub enum SubmitError {
    Busy,
    NotReady(String),
    Failed(String),
}

impl Engine {
    /// Load the tokenizer now; start the worker, which loads weights on its own thread.
    pub fn start(spec: Spec, cfg: EngineConfig) -> Result<Arc<Engine>, String> {
        let tokenizer = spec.tokenizer()?;
        let (tx, rx) = mpsc::sync_channel::<Job>(cfg.queue.max(1));
        let ready = Arc::new(AtomicBool::new(false));
        let failed = Arc::new(Mutex::new(None));
        let stats = Arc::new(Stats::default());
        let e = Arc::new(Engine {
            spec: spec.clone(),
            tokenizer,
            tx,
            ready: ready.clone(),
            failed: failed.clone(),
            stats: stats.clone(),
            device: cfg.device,
        });
        std::thread::Builder::new()
            .name(format!("inbed-{}", spec.id))
            .spawn(move || {
                let r = match cfg.device {
                    #[cfg(target_os = "macos")]
                    Device::Metal => crate::metal::Metal::new()
                        .and_then(|b| worker(b, &spec, cfg, rx, &ready, &stats)),
                    #[cfg(not(target_os = "macos"))]
                    Device::Metal => Err("Metal is only available on macOS".to_string()),
                    Device::Cpu => worker(crate::cpu::Cpu::new(), &spec, cfg, rx, &ready, &stats),
                };
                if let Err(e) = r {
                    eprintln!("inbed: model {} failed: {e}", spec.id);
                    *failed.lock().unwrap() = Some(e);
                }
            })
            .map_err(|e| e.to_string())?;
        Ok(e)
    }

    pub fn is_ready(&self) -> bool {
        self.ready.load(Ordering::Acquire)
    }

    pub fn failure(&self) -> Option<String> {
        self.failed.lock().unwrap().clone()
    }

    /// Block until the weights are loaded (or loading failed).
    pub fn wait_ready(&self) -> Result<(), String> {
        while !self.is_ready() {
            if let Some(e) = self.failure() {
                return Err(e);
            }
            std::thread::sleep(std::time::Duration::from_millis(20));
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

    /// Queue sequences; the receiver yields one full-dims, normalized vector per sequence.
    pub fn submit(&self, seqs: Vec<Vec<u32>>) -> Result<Receiver<Reply>, SubmitError> {
        if let Some(e) = self.failure() {
            return Err(SubmitError::Failed(e));
        }
        if !self.is_ready() {
            return Err(SubmitError::NotReady(format!(
                "model {} is still loading",
                self.spec.id
            )));
        }
        let (rtx, rrx) = mpsc::channel();
        match self.tx.try_send(Job { seqs, reply: rtx }) {
            Ok(()) => Ok(rrx),
            Err(TrySendError::Full(_)) => Err(SubmitError::Busy),
            Err(TrySendError::Disconnected(_)) => {
                Err(SubmitError::Failed("model worker stopped".into()))
            }
        }
    }

    /// Submit and wait.
    pub fn embed(&self, seqs: Vec<Vec<u32>>) -> Result<Vec<Vec<f32>>, String> {
        match self.submit(seqs) {
            Ok(rx) => rx.recv().map_err(|_| "model worker stopped".to_string())?,
            Err(SubmitError::Busy) => Err("queue full".into()),
            Err(SubmitError::NotReady(e)) | Err(SubmitError::Failed(e)) => Err(e),
        }
    }
}

fn worker<B: Backend>(
    mut b: B,
    spec: &Spec,
    cfg: EngineConfig,
    rx: Receiver<Job>,
    ready: &AtomicBool,
    stats: &Stats,
) -> Result<(), String> {
    let t0 = Instant::now();
    let w = model::load_weights(spec, &mut b)?;
    // warm up: first dispatches pay pipeline / allocation costs
    let _ = model::forward(spec, &w, &mut b, &Batch::new(&[&[0, 1, 2, 3]]));
    eprintln!(
        "inbed: {} ready on {} ({:.1}s)",
        spec.id,
        b.name(),
        t0.elapsed().as_secs_f64()
    );
    ready.store(true, Ordering::Release);

    while let Ok(first) = rx.recv() {
        // drain whatever else is waiting, up to a token budget, into one round
        let mut jobs = vec![first];
        let mut toks: usize = jobs[0].seqs.iter().map(Vec::len).sum();
        while toks < cfg.max_batch_tokens {
            match rx.try_recv() {
                Ok(j) => {
                    toks += j.seqs.iter().map(Vec::len).sum::<usize>();
                    jobs.push(j);
                }
                Err(_) => break,
            }
        }
        let started = Instant::now();
        // flatten, sort by length so each packed batch holds similar lengths, chunk by budget
        let mut all: Vec<(usize, usize, &[u32])> = jobs
            .iter()
            .enumerate()
            .flat_map(|(ji, j)| {
                j.seqs
                    .iter()
                    .enumerate()
                    .map(move |(si, s)| (ji, si, s.as_slice()))
            })
            .collect();
        all.sort_by_key(|x| x.2.len());
        let mut results: Vec<Vec<Vec<f32>>> = jobs
            .iter()
            .map(|j| vec![Vec::new(); j.seqs.len()])
            .collect();
        let mut i = 0;
        let mut batches = 0;
        let long_seq = crate::backend::long_seq();
        let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            while i < all.len() {
                let mut j = i;
                let mut t = 0;
                let long = all[i].2.len() >= long_seq;
                while j < all.len()
                    && (j == i
                        || (t + all[j].2.len() <= cfg.max_batch_tokens
                            && (all[j].2.len() >= long_seq) == long))
                {
                    t += all[j].2.len();
                    j += 1;
                }
                let seqs: Vec<&[u32]> = all[i..j].iter().map(|x| x.2).collect();
                let out = model::forward(spec, &w, &mut b, &Batch::new(&seqs));
                for (k, v) in out.into_iter().enumerate() {
                    let (ji, si, _) = all[i + k];
                    results[ji][si] = v;
                }
                batches += 1;
                i = j;
            }
        }));
        stats.batches.fetch_add(batches, Ordering::Relaxed);
        stats
            .busy_us
            .fetch_add(started.elapsed().as_micros() as u64, Ordering::Relaxed);
        stats.tokens.fetch_add(toks as u64, Ordering::Relaxed);
        stats
            .sequences
            .fetch_add(all.len() as u64, Ordering::Relaxed);
        stats
            .requests
            .fetch_add(jobs.len() as u64, Ordering::Relaxed);
        match r {
            Ok(()) => {
                for (j, res) in jobs.into_iter().zip(results) {
                    let _ = j.reply.send(Ok(res));
                }
            }
            Err(p) => {
                let msg = p
                    .downcast_ref::<String>()
                    .cloned()
                    .or_else(|| p.downcast_ref::<&str>().map(|s| s.to_string()))
                    .unwrap_or_else(|| "forward pass panicked".into());
                eprintln!("inbed: forward failed: {msg}");
                for j in jobs {
                    let _ = j.reply.send(Err(msg.clone()));
                }
            }
        }
    }
    Ok(())
}
