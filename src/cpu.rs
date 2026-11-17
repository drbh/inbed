//! f32 reference backend: simple, threaded, the oracle for GPU tests.

use crate::backend::{Act, AttnShape, Backend, Batch, WeightKind};

pub struct Cpu {
    batch: Batch,
    threads: usize,
}

#[derive(Clone)]
pub struct CpuW {
    pub data: Vec<f32>,
    pub rows: usize,
    pub cols: usize,
}

#[derive(Clone)]
pub struct CpuT {
    pub data: std::cell::RefCell<Vec<f32>>,
    pub rows: usize,
    pub cols: usize,
}

impl CpuT {
    fn new(rows: usize, cols: usize, data: Vec<f32>) -> CpuT {
        debug_assert_eq!(data.len(), rows * cols);
        CpuT {
            data: std::cell::RefCell::new(data),
            rows,
            cols,
        }
    }
}

impl Default for Cpu {
    fn default() -> Self {
        Cpu::new()
    }
}

impl Cpu {
    pub fn new() -> Cpu {
        Cpu {
            batch: Batch::default(),
            threads: std::thread::available_parallelism()
                .map(|n| n.get())
                .unwrap_or(4),
        }
    }
}

#[inline]
fn dot(a: &[f32], b: &[f32]) -> f32 {
    let mut acc = [0f32; 8];
    let (ac, bc) = (a.chunks_exact(8), b.chunks_exact(8));
    let (ar, br) = (ac.remainder(), bc.remainder());
    for (x, y) in ac.zip(bc) {
        for i in 0..8 {
            acc[i] += x[i] * y[i];
        }
    }
    let mut s: f32 = acc.iter().sum();
    for (x, y) in ar.iter().zip(br) {
        s += x * y;
    }
    s
}

/// Abramowitz–Stegun 7.1.26 (|error| < 1.5e-7). Shared with the Metal kernel.
pub fn erf(x: f32) -> f32 {
    let s = x.signum();
    let x = x.abs() as f64;
    let t = 1.0 / (1.0 + 0.3275911 * x);
    let y = 1.0
        - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t
            + 0.254829592)
            * t
            * (-x * x).exp();
    s * y as f32
}

pub fn gelu(x: f32) -> f32 {
    0.5 * x * (1.0 + erf(x / std::f32::consts::SQRT_2))
}

fn par_rows<F: Fn(usize, &mut [f32]) + Sync>(threads: usize, out: &mut [f32], cols: usize, f: F) {
    let rows = out.len() / cols.max(1);
    if rows == 0 {
        return;
    }
    let per = rows.div_ceil(threads).max(1);
    std::thread::scope(|s| {
        for (ci, chunk) in out.chunks_mut(per * cols).enumerate() {
            let f = &f;
            s.spawn(move || {
                for (r, row) in chunk.chunks_mut(cols).enumerate() {
                    f(ci * per + r, row);
                }
            });
        }
    });
}

impl Backend for Cpu {
    type W = CpuW;
    type T = CpuT;

    fn name(&self) -> &'static str {
        "cpu"
    }

    fn upload(&mut self, data: &[f32], rows: usize, cols: usize, _kind: WeightKind) -> CpuW {
        assert_eq!(data.len(), rows * cols);
        CpuW {
            data: data.to_vec(),
            rows,
            cols,
        }
    }

    fn begin(&mut self, batch: &Batch) {
        self.batch = batch.clone();
    }

    fn embed(&mut self, word: &CpuW, pos: Option<&CpuW>, type0: Option<&CpuW>) -> CpuT {
        let c = word.cols;
        let mut out = vec![0f32; self.batch.tokens() * c];
        for (t, row) in out.chunks_mut(c).enumerate() {
            let id = self.batch.ids[t] as usize;
            row.copy_from_slice(&word.data[id * c..(id + 1) * c]);
            if let Some(p) = pos {
                let pi = self.batch.positions[t] as usize;
                for (o, v) in row.iter_mut().zip(&p.data[pi * c..(pi + 1) * c]) {
                    *o += v;
                }
            }
            if let Some(ty) = type0 {
                for (o, v) in row.iter_mut().zip(&ty.data[..c]) {
                    *o += v;
                }
            }
        }
        CpuT::new(self.batch.tokens(), c, out)
    }

    fn linear(
        &mut self,
        x: &CpuT,
        w: &CpuW,
        b: Option<&CpuW>,
        act: Act,
        residual: Option<&CpuT>,
    ) -> CpuT {
        assert_eq!(x.cols, w.cols, "linear: in features");
        let (m, n, k) = (x.rows, w.rows, w.cols);
        let xg = x.data.borrow();
        let xd: &[f32] = &xg;
        let rd = residual.map(|r| r.data.borrow());
        let mut out = vec![0f32; m * n];
        // split output columns across threads (good for small m), each thread covers all rows
        let per = n.div_ceil(self.threads).max(16);
        let chunks: Vec<(usize, Vec<f32>)> = std::thread::scope(|s| {
            let hs: Vec<_> = (0..n)
                .step_by(per)
                .map(|n0| {
                    s.spawn(move || {
                        let n1 = (n0 + per).min(n);
                        let mut part = vec![0f32; m * (n1 - n0)];
                        for i in 0..m {
                            let xr = &xd[i * k..(i + 1) * k];
                            for j in n0..n1 {
                                part[i * (n1 - n0) + j - n0] = dot(xr, &w.data[j * k..(j + 1) * k]);
                            }
                        }
                        (n0, part)
                    })
                })
                .collect();
            hs.into_iter().map(|h| h.join().unwrap()).collect()
        });
        for (n0, part) in chunks {
            let width = part.len() / m.max(1);
            for i in 0..m {
                out[i * n + n0..i * n + n0 + width]
                    .copy_from_slice(&part[i * width..(i + 1) * width]);
            }
        }
        for i in 0..m {
            for j in 0..n {
                let mut v = out[i * n + j];
                if let Some(b) = b {
                    v += b.data[j];
                }
                if act == Act::Gelu {
                    v = gelu(v);
                }
                if let Some(r) = &rd {
                    v += r[i * n + j];
                }
                out[i * n + j] = v;
            }
        }
        CpuT::new(m, n, out)
    }

    fn layer_norm(&mut self, x: &CpuT, g: &CpuW, b: &CpuW, eps: f32) -> CpuT {
        let c = x.cols;
        let mut out = x.data.borrow().clone();
        for row in out.chunks_mut(c) {
            let mean = row.iter().sum::<f32>() / c as f32;
            let var = row.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / c as f32;
            let r = 1.0 / (var + eps).sqrt();
            for (j, v) in row.iter_mut().enumerate() {
                *v = (*v - mean) * r * g.data[j] + b.data[j];
            }
        }
        CpuT::new(x.rows, c, out)
    }

    fn rms_norm(&mut self, x: &CpuT, g: &CpuW, eps: f32) -> CpuT {
        let c = x.cols;
        let mut out = x.data.borrow().clone();
        for row in out.chunks_mut(c) {
            let ms = row.iter().map(|v| v * v).sum::<f32>() / c as f32;
            let r = 1.0 / (ms + eps).sqrt();
            for (j, v) in row.iter_mut().enumerate() {
                *v = *v * r * g.data[j];
            }
        }
        CpuT::new(x.rows, c, out)
    }

    fn qk_norm_rope(
        &mut self,
        qkv: &CpuT,
        s: &AttnShape,
        qn: &CpuW,
        kn: &CpuW,
        eps: f32,
        theta: f32,
    ) {
        let hd = s.head_dim;
        let half = hd / 2;
        let inv: Vec<f32> = (0..half)
            .map(|i| 1.0 / theta.powf((2 * i) as f32 / hd as f32))
            .collect();
        let mut d = qkv.data.borrow_mut();
        for (t, row) in d.chunks_mut(qkv.cols).enumerate() {
            let pos = self.batch.positions[t] as f32;
            for h in 0..s.n_q + s.n_kv {
                let w = if h < s.n_q { &qn.data } else { &kn.data };
                let x = &mut row[h * hd..(h + 1) * hd];
                let ms = x.iter().map(|v| v * v).sum::<f32>() / hd as f32;
                let r = 1.0 / (ms + eps).sqrt();
                for (j, v) in x.iter_mut().enumerate() {
                    *v = *v * r * w[j];
                }
                for i in 0..half {
                    let (sn, cs) = (pos * inv[i]).sin_cos();
                    let (a, b) = (x[i], x[i + half]);
                    x[i] = a * cs - b * sn;
                    x[i + half] = b * cs + a * sn;
                }
            }
        }
    }

    fn attention(&mut self, qkv: &CpuT, s: &AttnShape) -> CpuT {
        let hd = s.head_dim;
        let stride = qkv.cols;
        assert_eq!(stride, s.qkv_cols());
        let t_n = qkv.rows;
        let oc = s.n_q * hd;
        let scale = 1.0 / (hd as f32).sqrt();
        let group = s.n_q / s.n_kv;
        let dg = qkv.data.borrow();
        let d: &[f32] = &dg;
        let batch = &self.batch;
        let mut out = vec![0f32; t_n * oc];
        par_rows(self.threads, &mut out, oc, |t, orow| {
            let start = t - batch.positions[t] as usize;
            let end = if s.causal {
                t + 1
            } else {
                batch.seq_end[t] as usize
            };
            let mut sc = vec![0f32; end - start];
            for h in 0..s.n_q {
                let kvh = h / group;
                let q = &d[t * stride + h * hd..t * stride + (h + 1) * hd];
                for (jj, j) in (start..end).enumerate() {
                    let k =
                        &d[j * stride + (s.n_q + kvh) * hd..j * stride + (s.n_q + kvh + 1) * hd];
                    sc[jj] = dot(q, k) * scale;
                }
                let m = sc.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let mut sum = 0.0;
                for v in sc.iter_mut() {
                    *v = (*v - m).exp();
                    sum += *v;
                }
                let o = &mut orow[h * hd..(h + 1) * hd];
                for (jj, j) in (start..end).enumerate() {
                    let p = sc[jj] / sum;
                    let v = &d[j * stride + (s.n_q + s.n_kv + kvh) * hd
                        ..j * stride + (s.n_q + s.n_kv + kvh + 1) * hd];
                    for (oo, vv) in o.iter_mut().zip(v) {
                        *oo += p * vv;
                    }
                }
            }
        });
        CpuT::new(t_n, oc, out)
    }

    fn silu_mul(&mut self, x: &CpuT) -> CpuT {
        let i = x.cols / 2;
        let d = x.data.borrow();
        let mut out = vec![0f32; x.rows * i];
        for (r, row) in out.chunks_mut(i).enumerate() {
            let src = &d[r * x.cols..(r + 1) * x.cols];
            for j in 0..i {
                let g = src[j];
                row[j] = g / (1.0 + (-g).exp()) * src[i + j];
            }
        }
        CpuT::new(x.rows, i, out)
    }

    fn finish<R>(&mut self, x: &CpuT, f: impl FnOnce(&[f32], usize) -> R) -> R {
        f(&x.data.borrow(), x.cols)
    }
}
