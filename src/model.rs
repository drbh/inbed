//! Model specs (from a Hugging Face snapshot dir), weights, and the forward passes.

use crate::backend::{Act, AttnShape, Backend, Batch, WeightKind};
use crate::json::Json;
use crate::safetensors::SafeTensors;
use crate::tokenizer::Tokenizer;
use std::path::{Path, PathBuf};

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Pooling {
    Mean,
    Cls,
    Last,
}

#[derive(Clone, Debug)]
pub enum Arch {
    Bert {
        hidden: usize,
        layers: usize,
        heads: usize,
        intermediate: usize,
        eps: f32,
    },
    Qwen3 {
        hidden: usize,
        layers: usize,
        heads: usize,
        kv_heads: usize,
        head_dim: usize,
        intermediate: usize,
        eps: f32,
        theta: f32,
    },
}

impl Arch {
    pub fn hidden(&self) -> usize {
        match self {
            Arch::Bert { hidden, .. } | Arch::Qwen3 { hidden, .. } => *hidden,
        }
    }
    pub fn name(&self) -> &'static str {
        match self {
            Arch::Bert { .. } => "bert",
            Arch::Qwen3 { .. } => "qwen3",
        }
    }
}

/// Everything about a model except its weights.
#[derive(Clone, Debug)]
pub struct Spec {
    /// the name clients use (`model` in requests)
    pub id: String,
    pub dir: PathBuf,
    pub arch: Arch,
    pub pooling: Pooling,
    pub dims: usize,
    pub allowed_dims: Vec<usize>,
    pub max_tokens: usize,
    pub query_prefix: String,
    pub document_prefix: String,
    /// embedding table rows: token ids must be below this
    pub vocab: usize,
}

fn read_json(p: &Path) -> Result<Json, String> {
    let s = std::fs::read_to_string(p).map_err(|e| format!("{}: {e}", p.display()))?;
    Json::parse(&s).map_err(|e| format!("{}: {e}", p.display()))
}

fn opt_json(p: &Path) -> Option<Json> {
    std::fs::read_to_string(p)
        .ok()
        .and_then(|s| Json::parse(&s).ok())
}

fn num(j: &Json, k: &str) -> Result<usize, String> {
    j.get(k)
        .and_then(Json::as_u64)
        .map(|x| x as usize)
        .ok_or_else(|| format!("config.json: missing {k}"))
}

impl Spec {
    pub fn load(id: &str, dir: &Path, allowed_dims: Option<&[usize]>) -> Result<Spec, String> {
        let cfg = read_json(&dir.join("config.json"))?;
        let mt = cfg.get("model_type").and_then(Json::as_str).unwrap_or("");
        let arch = match mt {
            "bert" => {
                if cfg
                    .get("position_embedding_type")
                    .and_then(Json::as_str)
                    .is_some_and(|p| p != "absolute")
                {
                    return Err("only absolute position embeddings are supported for BERT".into());
                }
                if cfg
                    .get("hidden_act")
                    .and_then(Json::as_str)
                    .is_some_and(|a| a != "gelu")
                {
                    return Err("only gelu BERT models are supported".into());
                }
                Arch::Bert {
                    hidden: num(&cfg, "hidden_size")?,
                    layers: num(&cfg, "num_hidden_layers")?,
                    heads: num(&cfg, "num_attention_heads")?,
                    intermediate: num(&cfg, "intermediate_size")?,
                    eps: cfg
                        .get("layer_norm_eps")
                        .and_then(Json::as_f64)
                        .unwrap_or(1e-12) as f32,
                }
            }
            "qwen3" => Arch::Qwen3 {
                hidden: num(&cfg, "hidden_size")?,
                layers: num(&cfg, "num_hidden_layers")?,
                heads: num(&cfg, "num_attention_heads")?,
                kv_heads: num(&cfg, "num_key_value_heads")?,
                head_dim: cfg
                    .get("head_dim")
                    .and_then(Json::as_u64)
                    .map(|x| x as usize)
                    .unwrap_or(num(&cfg, "hidden_size")? / num(&cfg, "num_attention_heads")?),
                intermediate: num(&cfg, "intermediate_size")?,
                eps: cfg
                    .get("rms_norm_eps")
                    .and_then(Json::as_f64)
                    .unwrap_or(1e-6) as f32,
                theta: cfg.get("rope_theta").and_then(Json::as_f64).unwrap_or(1e6) as f32,
            },
            other => {
                return Err(format!(
                    "unsupported model_type {other:?} (supported: bert, qwen3)"
                ));
            }
        };
        let pool_cfg = opt_json(&dir.join("1_Pooling/config.json"));
        let flag = |k: &str| {
            pool_cfg
                .as_ref()
                .and_then(|p| p.get(k))
                .and_then(Json::as_bool)
                .unwrap_or(false)
        };
        let pooling = if flag("pooling_mode_lasttoken") {
            Pooling::Last
        } else if flag("pooling_mode_cls_token") {
            Pooling::Cls
        } else if flag("pooling_mode_mean_tokens") {
            Pooling::Mean
        } else {
            match arch {
                Arch::Bert { .. } => Pooling::Mean,
                Arch::Qwen3 { .. } => Pooling::Last,
            }
        };
        let max_pos = cfg
            .get("max_position_embeddings")
            .and_then(Json::as_u64)
            .unwrap_or(512) as usize;
        let max_tokens = opt_json(&dir.join("sentence_bert_config.json"))
            .and_then(|j| j.get("max_seq_length").and_then(Json::as_u64))
            .map(|x| x as usize)
            .unwrap_or(max_pos)
            .min(max_pos);
        let st = opt_json(&dir.join("config_sentence_transformers.json"));
        let prompt = |k: &str| {
            st.as_ref()
                .and_then(|j| j.get("prompts"))
                .and_then(|p| p.get(k))
                .and_then(Json::as_str)
                .unwrap_or("")
                .to_string()
        };
        let dims = arch.hidden();
        let allowed_dims = allowed_dims
            .map(|a| a.to_vec())
            .unwrap_or_else(|| vec![dims]);
        if allowed_dims.iter().any(|&d| d > dims || d == 0) {
            return Err(format!(
                "allowed dims {allowed_dims:?} exceed the model's {dims}"
            ));
        }
        Ok(Spec {
            id: id.to_string(),
            dir: dir.to_path_buf(),
            arch,
            pooling,
            dims,
            allowed_dims,
            max_tokens,
            vocab: num(&cfg, "vocab_size")?,
            query_prefix: prompt("query"),
            document_prefix: prompt("document"),
        })
    }

    pub fn tokenizer(&self) -> Result<Tokenizer, String> {
        Tokenizer::from_file(&self.dir.join("tokenizer.json"))
    }
}

// ------------------------------------------------------------------ weights

pub struct BertLayer<W> {
    qkv_w: W,
    qkv_b: W,
    o_w: W,
    o_b: W,
    ln1_g: W,
    ln1_b: W,
    i_w: W,
    i_b: W,
    out_w: W,
    out_b: W,
    ln2_g: W,
    ln2_b: W,
}

pub struct QwenLayer<W> {
    in_ln: W,
    qkv_w: W,
    q_norm: W,
    k_norm: W,
    o_w: W,
    post_ln: W,
    gu_w: W,
    down_w: W,
}

pub enum Weights<W> {
    Bert {
        word: W,
        pos: W,
        typ: W,
        ln_g: W,
        ln_b: W,
        layers: Vec<BertLayer<W>>,
    },
    Qwen3 {
        embed: W,
        layers: Vec<QwenLayer<W>>,
        norm: W,
    },
}

struct Loader<'a, B: Backend> {
    st: &'a SafeTensors,
    b: &'a mut B,
    prefix: String,
}

impl<B: Backend> Loader<'_, B> {
    fn name(&self, n: &str) -> String {
        if self.st.tensors.contains_key(n) {
            n.to_string()
        } else {
            format!("{}{n}", self.prefix)
        }
    }
    fn raw(&self, n: &str) -> Result<(Vec<f32>, Vec<usize>), String> {
        self.st.f32(&self.name(n))
    }
    fn get(&mut self, n: &str, kind: WeightKind) -> Result<B::W, String> {
        let (v, shape) = self.raw(n)?;
        let (r, c) = match shape.len() {
            1 => (1, shape[0]),
            2 => (shape[0], shape[1]),
            _ => return Err(format!("{n}: expected 1-D or 2-D")),
        };
        Ok(self.b.upload(&v, r, c, kind))
    }
    /// Concatenate matrices (or vectors) along rows.
    fn cat(&mut self, names: &[String], kind: WeightKind) -> Result<B::W, String> {
        let mut all = Vec::new();
        let mut rows = 0;
        let mut cols = 0;
        for n in names {
            let (v, shape) = self.raw(n)?;
            let (r, c) = if shape.len() == 1 {
                (shape[0], 1)
            } else {
                (shape[0], shape[1])
            };
            if cols != 0 && c != cols {
                return Err(format!("{n}: column mismatch"));
            }
            cols = c;
            rows += r;
            all.extend(v);
        }
        Ok(if kind == WeightKind::Vector {
            self.b.upload(&all, 1, rows, kind)
        } else {
            self.b.upload(&all, rows, cols, kind)
        })
    }
}

pub fn load_weights<B: Backend>(spec: &Spec, b: &mut B) -> Result<Weights<B::W>, String> {
    let st = SafeTensors::open(&spec.dir.join("model.safetensors"))?;
    use WeightKind::*;
    match spec.arch {
        Arch::Bert { layers, .. } => {
            let prefix = if st.tensors.contains_key("embeddings.word_embeddings.weight") {
                ""
            } else {
                "bert."
            }
            .to_string();
            let mut l = Loader { st: &st, b, prefix };
            let word = l.get("embeddings.word_embeddings.weight", Table)?;
            let pos = l.get("embeddings.position_embeddings.weight", Table)?;
            let typ = l.get("embeddings.token_type_embeddings.weight", Table)?;
            let ln_g = l.get("embeddings.LayerNorm.weight", Vector)?;
            let ln_b = l.get("embeddings.LayerNorm.bias", Vector)?;
            let mut ls = Vec::new();
            for i in 0..layers {
                let p = format!("encoder.layer.{i}.");
                let n = |s: &str| format!("{p}{s}");
                ls.push(BertLayer {
                    qkv_w: l.cat(
                        &[
                            n("attention.self.query.weight"),
                            n("attention.self.key.weight"),
                            n("attention.self.value.weight"),
                        ],
                        Matrix,
                    )?,
                    qkv_b: l.cat(
                        &[
                            n("attention.self.query.bias"),
                            n("attention.self.key.bias"),
                            n("attention.self.value.bias"),
                        ],
                        Vector,
                    )?,
                    o_w: l.get(&n("attention.output.dense.weight"), Matrix)?,
                    o_b: l.get(&n("attention.output.dense.bias"), Vector)?,
                    ln1_g: l.get(&n("attention.output.LayerNorm.weight"), Vector)?,
                    ln1_b: l.get(&n("attention.output.LayerNorm.bias"), Vector)?,
                    i_w: l.get(&n("intermediate.dense.weight"), Matrix)?,
                    i_b: l.get(&n("intermediate.dense.bias"), Vector)?,
                    out_w: l.get(&n("output.dense.weight"), Matrix)?,
                    out_b: l.get(&n("output.dense.bias"), Vector)?,
                    ln2_g: l.get(&n("output.LayerNorm.weight"), Vector)?,
                    ln2_b: l.get(&n("output.LayerNorm.bias"), Vector)?,
                });
            }
            Ok(Weights::Bert {
                word,
                pos,
                typ,
                ln_g,
                ln_b,
                layers: ls,
            })
        }
        Arch::Qwen3 { layers, .. } => {
            let prefix = if st.tensors.contains_key("embed_tokens.weight") {
                ""
            } else {
                "model."
            }
            .to_string();
            let mut l = Loader { st: &st, b, prefix };
            let embed = l.get("embed_tokens.weight", Table)?;
            let mut ls = Vec::new();
            for i in 0..layers {
                let p = format!("layers.{i}.");
                let n = |s: &str| format!("{p}{s}");
                ls.push(QwenLayer {
                    in_ln: l.get(&n("input_layernorm.weight"), Vector)?,
                    qkv_w: l.cat(
                        &[
                            n("self_attn.q_proj.weight"),
                            n("self_attn.k_proj.weight"),
                            n("self_attn.v_proj.weight"),
                        ],
                        Matrix,
                    )?,
                    q_norm: l.get(&n("self_attn.q_norm.weight"), Vector)?,
                    k_norm: l.get(&n("self_attn.k_norm.weight"), Vector)?,
                    o_w: l.get(&n("self_attn.o_proj.weight"), Matrix)?,
                    post_ln: l.get(&n("post_attention_layernorm.weight"), Vector)?,
                    gu_w: l.cat(
                        &[n("mlp.gate_proj.weight"), n("mlp.up_proj.weight")],
                        Matrix,
                    )?,
                    down_w: l.get(&n("mlp.down_proj.weight"), Matrix)?,
                });
            }
            let norm = l.get("norm.weight", Vector)?;
            Ok(Weights::Qwen3 {
                embed,
                layers: ls,
                norm,
            })
        }
    }
}

// ------------------------------------------------------------------ forward

/// Run a packed batch and return one pooled, L2-normalized vector (full dims) per sequence.
pub fn forward<B: Backend>(
    spec: &Spec,
    w: &Weights<B::W>,
    b: &mut B,
    batch: &Batch,
) -> Vec<Vec<f32>> {
    if batch.tokens() == 0 {
        return vec![];
    }
    b.begin(batch);
    let x = match (&spec.arch, w) {
        (
            &Arch::Bert {
                hidden, heads, eps, ..
            },
            Weights::Bert {
                word,
                pos,
                typ,
                ln_g,
                ln_b,
                layers,
            },
        ) => {
            let s = AttnShape {
                n_q: heads,
                n_kv: heads,
                head_dim: hidden / heads,
                causal: false,
            };
            let e = b.embed(word, Some(pos), Some(typ));
            let mut x = b.layer_norm(&e, ln_g, ln_b, eps);
            for l in layers {
                let qkv = b.linear(&x, &l.qkv_w, Some(&l.qkv_b), Act::None, None);
                let a = b.attention(&qkv, &s);
                let h = b.linear(&a, &l.o_w, Some(&l.o_b), Act::None, Some(&x));
                let x1 = b.layer_norm(&h, &l.ln1_g, &l.ln1_b, eps);
                let i = b.linear(&x1, &l.i_w, Some(&l.i_b), Act::Gelu, None);
                let o = b.linear(&i, &l.out_w, Some(&l.out_b), Act::None, Some(&x1));
                x = b.layer_norm(&o, &l.ln2_g, &l.ln2_b, eps);
            }
            x
        }
        (
            &Arch::Qwen3 {
                heads,
                kv_heads,
                head_dim,
                eps,
                theta,
                ..
            },
            Weights::Qwen3 {
                embed,
                layers,
                norm,
            },
        ) => {
            let s = AttnShape {
                n_q: heads,
                n_kv: kv_heads,
                head_dim,
                causal: true,
            };
            let mut x = b.embed(embed, None, None);
            for l in layers {
                let h = b.rms_norm(&x, &l.in_ln, eps);
                let qkv = b.linear(&h, &l.qkv_w, None, Act::None, None);
                b.qk_norm_rope(&qkv, &s, &l.q_norm, &l.k_norm, eps, theta);
                let a = b.attention(&qkv, &s);
                x = b.linear(&a, &l.o_w, None, Act::None, Some(&x));
                let h = b.rms_norm(&x, &l.post_ln, eps);
                let gu = b.linear(&h, &l.gu_w, None, Act::None, None);
                let m = b.silu_mul(&gu);
                x = b.linear(&m, &l.down_w, None, Act::None, Some(&x));
            }
            b.rms_norm(&x, norm, eps)
        }
        _ => unreachable!("weights do not match arch"),
    };
    let pooling = spec.pooling;
    b.finish(&x, |data, cols| pool(data, cols, batch, pooling))
}

pub fn pool(data: &[f32], cols: usize, batch: &Batch, mode: Pooling) -> Vec<Vec<f32>> {
    (0..batch.seqs())
        .map(|i| {
            let (s, e) = (batch.starts[i] as usize, batch.starts[i + 1] as usize);
            let mut v = match mode {
                Pooling::Cls => data[s * cols..(s + 1) * cols].to_vec(),
                Pooling::Last => data[(e - 1) * cols..e * cols].to_vec(),
                Pooling::Mean => {
                    let mut acc = vec![0f64; cols];
                    for r in s..e {
                        for (a, x) in acc.iter_mut().zip(&data[r * cols..(r + 1) * cols]) {
                            *a += *x as f64;
                        }
                    }
                    acc.iter().map(|a| (a / (e - s) as f64) as f32).collect()
                }
            };
            normalize(&mut v);
            v
        })
        .collect()
}

pub fn normalize(v: &mut [f32]) {
    let n = v
        .iter()
        .map(|x| (*x as f64) * (*x as f64))
        .sum::<f64>()
        .sqrt();
    if n > 0.0 {
        for x in v.iter_mut() {
            *x = (*x as f64 / n) as f32;
        }
    }
}

/// Matryoshka truncation: first `dims` values, re-normalized.
pub fn truncate_dims(mut v: Vec<f32>, dims: usize) -> Vec<f32> {
    if dims < v.len() {
        v.truncate(dims);
        normalize(&mut v);
    }
    v
}
