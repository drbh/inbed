//! Engine scheduling: admission by expected wait, queries ahead of documents,
//! document requests split across slices. MiniLM on the CPU backend.

use inbed::engine::{Device, Engine, EngineConfig, Kind, SubmitError};
use std::sync::Arc;
use std::time::{Duration, Instant};

const MODEL: &str = "sentence-transformers/all-MiniLM-L6-v2";

fn engine(cfg: EngineConfig) -> Option<Arc<Engine>> {
    let r = inbed::hub::resolve(MODEL).ok()?;
    let spec = inbed::model::Spec::load(&r.id, &r.dir, None).unwrap();
    let e = Engine::start(spec, cfg).unwrap();
    e.wait_ready().unwrap();
    Some(e)
}

/// A document request of `n` 256-token texts.
fn docs(e: &Engine, n: usize) -> Vec<Vec<u32>> {
    let text = "the quick brown fox jumps over the lazy dog ".repeat(60);
    (0..n).map(|_| e.tokenize(&text, false)).collect()
}

#[test]
fn queries_jump_document_backlog() {
    let Some(e) = engine(EngineConfig {
        device: Device::Cpu,
        doc_slice_ms: 5,
        doc_max_wait_ms: 60_000,
        ..Default::default()
    }) else {
        eprintln!("skip: {MODEL} not cached");
        return;
    };
    let one = Instant::now();
    e.embed(docs(&e, 16)).unwrap();
    let per_request = one.elapsed();

    // a backlog of document requests, each submitted and waited on by its own thread
    let hs: Vec<_> = (0..8)
        .map(|_| {
            let e = e.clone();
            std::thread::spawn(move || {
                let seqs = docs(&e, 16);
                e.submit(seqs, Kind::Document).unwrap().wait().unwrap();
                Instant::now()
            })
        })
        .collect();
    while e.queued_tokens(Kind::Document) < 4 * 16 * 256 {
        std::thread::yield_now();
    }
    let t = Instant::now();
    let q = e
        .submit(vec![e.tokenize("red fox", true)], Kind::Query)
        .unwrap()
        .wait()
        .unwrap();
    let query_done = Instant::now();
    let latency = t.elapsed();
    let last_doc = hs.into_iter().map(|h| h.join().unwrap()).max().unwrap();
    assert_eq!(q.vecs.len(), 1);
    assert!(
        query_done < last_doc,
        "the query should finish before the document backlog"
    );
    // it waited behind at most a slice or two, not behind whole 16-text requests
    assert!(
        latency < per_request,
        "query took {latency:?}; one document request takes {per_request:?}"
    );
    eprintln!(
        "query under backlog: {latency:?} (queue {} µs, compute {} µs); one document request {per_request:?}",
        q.queue_us, q.compute_us
    );
}

#[test]
fn admission_rejects_over_wait_limit() {
    let Some(e) = engine(EngineConfig {
        device: Device::Cpu,
        doc_max_wait_ms: 1,
        ..Default::default()
    }) else {
        return;
    };
    // an empty queue always admits, even a request far over the limit
    let big = e.submit(docs(&e, 64), Kind::Document).unwrap();
    // the next one would wait behind it: rejected at once, with a retry hint
    let t = Instant::now();
    match e.submit(docs(&e, 1), Kind::Document) {
        Err(SubmitError::Busy { retry_after_ms }) => assert!(retry_after_ms > 0),
        other => panic!("expected Busy, got {:?}", other.map(|_| ())),
    }
    assert!(t.elapsed() < Duration::from_millis(50));
    assert_eq!(
        e.stats()
            .rejected
            .load(std::sync::atomic::Ordering::Relaxed),
        1
    );
    // queries have their own budget and are still admitted
    let q = e
        .submit(vec![e.tokenize("red fox", true)], Kind::Query)
        .unwrap();
    q.wait().unwrap();
    big.wait().unwrap();
    // everything drained: the counters return to zero
    assert_eq!(e.queued_tokens(Kind::Document), 0);
    assert_eq!(e.queued_tokens(Kind::Query), 0);
    e.submit(docs(&e, 1), Kind::Document)
        .unwrap()
        .wait()
        .unwrap();
}

#[test]
fn sliced_requests_match_single() {
    let Some(e) = engine(EngineConfig {
        device: Device::Cpu,
        doc_slice_ms: 1, // forces one or two texts per forward pass
        ..Default::default()
    }) else {
        return;
    };
    let texts = ["red fox", "blue bird", "Café crème", "中文测试", ""];
    let seqs: Vec<Vec<u32>> = texts.iter().map(|t| e.tokenize(t, false)).collect();
    let together = e.embed(seqs.clone()).unwrap();
    for (s, v) in seqs.into_iter().zip(&together) {
        let alone = e.embed(vec![s]).unwrap();
        let cos: f32 = alone[0].iter().zip(v).map(|(a, b)| a * b).sum();
        assert!(cos > 0.99999, "cos {cos}");
    }
}
