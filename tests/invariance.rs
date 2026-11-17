//! Batch invariance: does a text get the same vector whatever it is batched
//! with? The CPU backend must be bit-identical. On Metal, MPS picks kernels by matrix shape, so
//! the test reports how far apart the results are and holds them to a cosine bound.

use inbed::backend::{Backend, Batch};
use inbed::model::{self, Spec};

fn spec(name: &str) -> Option<Spec> {
    let r = inbed::hub::resolve(name).ok()?;
    Some(Spec::load(&r.id, &r.dir, r.allowed_dims.as_deref()).unwrap())
}

/// (bit-identical compositions, total compositions, max |Δ| over components, min cosine)
fn check<B: Backend>(name: &str, b: &mut B) -> Option<(usize, usize, f32, f32)> {
    let spec = spec(name)?;
    let tok = spec.tokenizer().unwrap();
    let w = model::load_weights(&spec, b).unwrap();
    let target = tok.encode(
        "Turbopuffer stores vectors in object storage.",
        spec.max_tokens,
    );
    let short = tok.encode("red fox", spec.max_tokens);
    let long = tok.encode(
        &"embeddings map text to vectors ".repeat(40),
        spec.max_tokens,
    );
    let alone = model::forward(&spec, &w, b, &Batch::new(&[&target])).remove(0);
    // vary what comes before and after the target, and the batch's total rows
    let mut comps: Vec<(Vec<&[u32]>, usize)> = vec![
        (vec![&short, &target], 1),
        (vec![&target, &long], 0),
        (vec![&long, &target, &short], 1),
    ];
    for n in [3, 17, 60] {
        let mut v: Vec<&[u32]> = (0..n)
            .map(|i| if i % 2 == 0 { &short[..] } else { &long[..] })
            .collect();
        v.push(&target);
        comps.push((v, n));
    }
    let (mut same, mut max_d, mut min_cos) = (0, 0f32, 1f32);
    for (seqs, at) in &comps {
        let got = model::forward(&spec, &w, b, &Batch::new(seqs)).remove(*at);
        if got == alone {
            same += 1;
        }
        let d = got
            .iter()
            .zip(&alone)
            .map(|(x, y)| (x - y).abs())
            .fold(0f32, f32::max);
        let c: f32 = got.iter().zip(&alone).map(|(x, y)| x * y).sum();
        max_d = max_d.max(d);
        min_cos = min_cos.min(c);
    }
    eprintln!(
        "{name} on {}: {same}/{} compositions bit-identical, max |Δ| {max_d:.2e}, min cos {min_cos:.9}",
        b.name(),
        comps.len()
    );
    Some((same, comps.len(), max_d, min_cos))
}

#[test]
fn cpu_is_batch_invariant() {
    let Some((same, n, _, _)) = check(
        "sentence-transformers/all-MiniLM-L6-v2",
        &mut inbed::cpu::Cpu::new(),
    ) else {
        return;
    };
    assert_eq!(
        same, n,
        "the CPU backend must be bit-identical across batches"
    );
}

#[cfg(target_os = "macos")]
#[test]
fn metal_batch_variance_is_bounded() {
    for m in [
        "sentence-transformers/all-MiniLM-L6-v2",
        "qwen/qwen3-embedding-0p6b",
    ] {
        let mut b = inbed::metal::Metal::new().unwrap();
        if let Some((_, _, _, min_cos)) = check(m, &mut b) {
            assert!(min_cos > 0.99999, "{m}: min cos {min_cos}");
        }
    }
}
