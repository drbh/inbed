//! The op set a model forward pass is written against. Implemented by the CPU reference and Metal.
//!
//! Activations are 2-D `[tokens, features]` over a *packed* varlen batch: every sequence's tokens are
//! concatenated, so matmuls never see padding. Only attention needs the sequence boundaries.

/// A packed batch of sequences.
#[derive(Clone, Debug, Default)]
pub struct Batch {
    pub ids: Vec<u32>,
    /// position of each token within its sequence (0-based)
    pub positions: Vec<u32>,
    /// exclusive end index (into `ids`) of each token's sequence
    pub seq_end: Vec<u32>,
    /// start offsets of each sequence, plus a final `ids.len()`
    pub starts: Vec<u32>,
}

impl Batch {
    pub fn new(seqs: &[&[u32]]) -> Batch {
        let mut b = Batch::default();
        for s in seqs {
            let start = b.ids.len() as u32;
            let end = start + s.len() as u32;
            b.starts.push(start);
            for (p, &id) in s.iter().enumerate() {
                b.ids.push(id);
                b.positions.push(p as u32);
                b.seq_end.push(end);
            }
        }
        b.starts.push(b.ids.len() as u32);
        b
    }
    pub fn tokens(&self) -> usize {
        self.ids.len()
    }
    pub fn seqs(&self) -> usize {
        self.starts.len() - 1
    }
}

/// Sequences at least this long are batched only with other long sequences, so a backend can
/// switch to an attention path suited to them (`INBED_LONG_ATTN`, default 1024).
pub fn long_seq() -> usize {
    std::env::var("INBED_LONG_ATTN")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(1024)
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Act {
    None,
    Gelu,
}

/// How a weight will be used, so a backend can pick its storage (e.g. f16 matrices on the GPU).
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum WeightKind {
    /// `[out, in]` matrix for `linear`
    Matrix,
    /// `[rows, cols]` table for embedding gathers
    Table,
    /// 1-D vector: biases and norm weights (kept f32)
    Vector,
}

pub struct AttnShape {
    pub n_q: usize,
    pub n_kv: usize,
    pub head_dim: usize,
    pub causal: bool,
}

impl AttnShape {
    /// columns of the fused `[q | k | v]` projection
    pub fn qkv_cols(&self) -> usize {
        (self.n_q + 2 * self.n_kv) * self.head_dim
    }
}

pub trait Backend {
    type W;
    type T;

    fn name(&self) -> &'static str;

    fn upload(&mut self, data: &[f32], rows: usize, cols: usize, kind: WeightKind) -> Self::W;

    /// Start a forward pass over `batch`.
    fn begin(&mut self, batch: &Batch);

    /// `out[t] = word[ids[t]] (+ pos[positions[t]]) (+ type0[0])`
    fn embed(&mut self, word: &Self::W, pos: Option<&Self::W>, type0: Option<&Self::W>) -> Self::T;

    /// `act(x · Wᵀ + b) + residual`
    fn linear(
        &mut self,
        x: &Self::T,
        w: &Self::W,
        b: Option<&Self::W>,
        act: Act,
        residual: Option<&Self::T>,
    ) -> Self::T;

    fn layer_norm(&mut self, x: &Self::T, g: &Self::W, b: &Self::W, eps: f32) -> Self::T;

    fn rms_norm(&mut self, x: &Self::T, g: &Self::W, eps: f32) -> Self::T;

    /// In place on the fused qkv tensor: per-head RMSNorm of q and k heads (Qwen3), then rotary
    /// embedding (rotate-half) at each token's position.
    fn qk_norm_rope(
        &mut self,
        qkv: &Self::T,
        s: &AttnShape,
        qn: &Self::W,
        kn: &Self::W,
        eps: f32,
        theta: f32,
    );

    /// Scaled dot-product attention within each sequence; input is the fused `[q|k|v]` tensor,
    /// output `[tokens, n_q * head_dim]`.
    fn attention(&mut self, qkv: &Self::T, s: &AttnShape) -> Self::T;

    /// `[t, 2I] → [t, I]`: `silu(x[:, :I]) * x[:, I:]`
    fn silu_mul(&mut self, x: &Self::T) -> Self::T;

    /// Finish the pass and hand the final `[tokens, cols]` activations to `f` (row-major).
    fn finish<R>(&mut self, x: &Self::T, f: impl FnOnce(&[f32], usize) -> R) -> R;
}
