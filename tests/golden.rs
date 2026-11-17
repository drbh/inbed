//! Compare against reference fixtures (token ids and vectors from sentence-transformers).
//!
//!     cargo test --release --test golden
//!
//! Tests for a model whose weights are not in the local Hugging Face cache are skipped.

use inbed::backend::{Backend, Batch};
use inbed::json::Json;
use inbed::model::{self, Spec};
use inbed::tokenizer::Tokenizer;

pub struct Case {
    pub text: String,
    pub ids: Vec<u32>,
    pub vec: Vec<f32>,
}

pub struct Fixture {
    pub max_len: usize,
    pub prefix: String,
    pub docs: Vec<Case>,
    pub queries: Vec<Case>,
}

pub fn fixture(name: &str) -> Fixture {
    let p = format!("{}/tests/fixtures/{name}.json", env!("CARGO_MANIFEST_DIR"));
    let j = Json::parse(&std::fs::read_to_string(p).unwrap()).unwrap();
    let cases = |k: &str| {
        j.get(k)
            .unwrap()
            .as_arr()
            .unwrap()
            .iter()
            .map(|c| Case {
                text: c.get("text").unwrap().as_str().unwrap().to_string(),
                ids: c
                    .get("ids")
                    .unwrap()
                    .as_arr()
                    .unwrap()
                    .iter()
                    .map(|x| x.as_u64().unwrap() as u32)
                    .collect(),
                vec: c
                    .get("vec")
                    .unwrap()
                    .as_arr()
                    .unwrap()
                    .iter()
                    .map(|x| x.as_f64().unwrap() as f32)
                    .collect(),
            })
            .collect()
    };
    Fixture {
        max_len: j.get("max_seq_length").unwrap().as_u64().unwrap() as usize,
        prefix: j.get("query_prefix").unwrap().as_str().unwrap().to_string(),
        docs: cases("docs"),
        queries: cases("queries"),
    }
}

pub fn spec(name: &str) -> Option<Spec> {
    let r = inbed::hub::resolve(name).ok()?;
    Some(Spec::load(&r.id, &r.dir, r.allowed_dims.as_deref()).unwrap())
}

pub fn cos(a: &[f32], b: &[f32]) -> f32 {
    let d: f64 = a.iter().zip(b).map(|(x, y)| *x as f64 * *y as f64).sum();
    let na: f64 = a.iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt();
    let nb: f64 = b.iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt();
    (d / (na * nb)) as f32
}

fn check_tokenizer(model: &str, fx: &str) {
    let Some(spec) = spec(model) else {
        eprintln!("skip: {model} not cached");
        return;
    };
    let tok: Tokenizer = spec.tokenizer().unwrap();
    let f = fixture(fx);
    let mut bad = 0;
    for c in &f.docs {
        let got = tok.encode(&c.text, f.max_len);
        if got != c.ids {
            bad += 1;
            eprintln!(
                "MISMATCH {:?}\n  want {:?}\n  got  {:?}",
                c.text, c.ids, got
            );
        }
    }
    for c in &f.queries {
        let got = tok.encode(&format!("{}{}", f.prefix, c.text), f.max_len);
        if got != c.ids {
            bad += 1;
            eprintln!(
                "MISMATCH query {:?}\n  want {:?}\n  got  {:?}",
                c.text, c.ids, got
            );
        }
    }
    assert_eq!(bad, 0, "{bad} tokenizer mismatches");
}

/// Embed every fixture input (one sequence per forward, then all in one packed batch) and compare.
pub fn check_vectors<B: Backend>(model: &str, fx: &str, b: &mut B, min_cos: f32) {
    let Some(spec) = spec(model) else {
        eprintln!("skip: {model} not cached");
        return;
    };
    let w = model::load_weights(&spec, b).unwrap();
    let f = fixture(fx);
    let cases: Vec<&Case> = f.docs.iter().chain(&f.queries).collect();
    let mut worst = 1.0f32;
    let mut singles = Vec::new();
    for c in &cases {
        let out = model::forward(&spec, &w, b, &Batch::new(&[&c.ids]));
        let cs = cos(&out[0], &c.vec);
        if cs < min_cos {
            eprintln!("LOW cos {cs:.6} for {:?}", c.text);
        }
        worst = worst.min(cs);
        singles.push(out.into_iter().next().unwrap());
    }
    eprintln!("{model} on {}: worst cos {worst:.7}", b.name());
    assert!(worst >= min_cos, "worst cosine {worst} < {min_cos}");

    // packed batch must equal one-at-a-time
    let seqs: Vec<&[u32]> = cases.iter().map(|c| c.ids.as_slice()).collect();
    let batched = model::forward(&spec, &w, b, &Batch::new(&seqs));
    for (i, (a, s)) in batched.iter().zip(&singles).enumerate() {
        let cs = cos(a, s);
        assert!(
            cs >= 0.99999,
            "batched != single for {:?}: cos {cs}",
            cases[i].text
        );
    }
}

#[test]
fn minilm_tokenizer() {
    check_tokenizer("sentence-transformers/all-MiniLM-L6-v2", "minilm");
}

#[test]
fn qwen3_tokenizer() {
    check_tokenizer("qwen/qwen3-embedding-0p6b", "qwen3");
}

#[test]
fn minilm_cpu() {
    check_vectors(
        "sentence-transformers/all-MiniLM-L6-v2",
        "minilm",
        &mut inbed::cpu::Cpu::new(),
        0.9999,
    );
}

#[test]
#[ignore = "slow on the CPU reference; run with --ignored"]
fn qwen3_cpu() {
    check_vectors(
        "qwen/qwen3-embedding-0p6b",
        "qwen3",
        &mut inbed::cpu::Cpu::new(),
        0.9999,
    );
}

#[cfg(target_os = "macos")]
#[test]
fn minilm_metal() {
    let mut m = inbed::metal::Metal::new().unwrap();
    check_vectors(
        "sentence-transformers/all-MiniLM-L6-v2",
        "minilm",
        &mut m,
        0.999,
    );
}

#[cfg(target_os = "macos")]
#[test]
fn qwen3_metal() {
    let mut m = inbed::metal::Metal::new().unwrap();
    check_vectors("qwen/qwen3-embedding-0p6b", "qwen3", &mut m, 0.999);
}
