//! Metal backend: one command buffer per forward pass, f16 weights, f32 activations.

pub(crate) mod objc;

#[link(name = "MetalPerformanceShaders", kind = "framework")]
unsafe extern "C" {}

use crate::backend::{Act, AttnShape, Backend, Batch, WeightKind};
use crate::msg;
use crate::safetensors::f32_to_f16;
use objc::{AutoreleasePool, Id, MTLSize, size};
use std::cell::RefCell;
use std::collections::HashMap;
use std::ffi::c_void;
use std::rc::Rc;

const SOURCE: &str = include_str!("kernels.metal");
const KERNELS: &[&str] = &[
    "gemm",
    "epilogue",
    "embed",
    "layer_norm",
    "rms_norm",
    "qk_norm_rope",
    "attention",
    "flash_attention_32",
    "flash_attention_64",
    "flash_attention_128",
    "softmax_rows",
    "silu_mul",
];

/// Shared-storage MTLBuffer. Pooled buffers go back to their pool on drop.
pub struct Buf {
    id: Id,
    ptr: *mut u8,
    bytes: usize,
    pool: Option<Rc<RefCell<Pool>>>,
}

impl Drop for Buf {
    fn drop(&mut self) {
        match &self.pool {
            Some(p) => p
                .borrow_mut()
                .free
                .entry(self.bytes)
                .or_default()
                .push((self.id, self.ptr)),
            None => objc::release(self.id),
        }
    }
}

#[derive(Default)]
pub struct Pool {
    free: HashMap<usize, Vec<(Id, *mut u8)>>,
    allocated: usize,
}

impl Drop for Pool {
    fn drop(&mut self) {
        for v in self.free.values() {
            for (id, _) in v {
                objc::release(*id);
            }
        }
    }
}

pub struct MetalW {
    buf: Buf,
    rows: usize,
    cols: usize,
}

pub struct MetalT {
    buf: Buf,
    rows: usize,
    cols: usize,
}

struct Pass {
    _pool: AutoreleasePool,
    cb: std::cell::Cell<Id>,
    enc: std::cell::Cell<Id>,
    ids: Buf,
    pos: Buf,
    seq_end: Buf,
}

pub struct Metal {
    dev: Id,
    queue: Id,
    psos: HashMap<&'static str, Id>,
    pool: Rc<RefCell<Pool>>,
    dummy: Buf,
    pass: Option<Pass>,
    tokens: usize,
    device_name: String,
    /// matmuls via MetalPerformanceShaders (default) or the hand-written `gemm` kernel
    use_mps: bool,
    /// tiled simdgroup-MMA attention (default) vs the one-simdgroup-per-row kernel (INBED_ATTN=simple)
    flash: bool,
    /// MPSMatrixMultiplication kernels by (m, n, k)
    mps: RefCell<HashMap<(usize, usize, usize, bool), Id>>,
    /// sequence starts of the current batch (+ total)
    starts: Vec<u32>,
    /// batches whose sequences are all at least this long use matmul attention
    long_attn: usize,
    /// INBED_TIMING=1: print encode / GPU time per pass
    timing: bool,
    pass_started: Option<std::time::Instant>,
    /// INBED_PROFILE=1: one command buffer per dispatch, GPU time per kernel
    profile: Option<RefCell<HashMap<&'static str, (u64, f64)>>>,
}

impl Drop for Metal {
    fn drop(&mut self) {
        self.pass = None;
        for k in self.mps.borrow().values() {
            objc::release(*k);
        }
        for p in self.psos.values() {
            objc::release(*p);
        }
        objc::release(self.queue);
        objc::release(self.dev);
    }
}

#[repr(C)]
struct GemmP {
    m: u32,
    n: u32,
    k: u32,
    has_bias: u32,
    has_res: u32,
    act: u32,
}
#[repr(C)]
struct EmbedP {
    cols: u32,
    has_pos: u32,
    has_type: u32,
}
#[repr(C)]
struct NormP {
    cols: u32,
    eps: f32,
}
#[repr(C)]
struct RopeP {
    stride: u32,
    n_q: u32,
    n_kv: u32,
    hd: u32,
    eps: f32,
    theta: f32,
}
#[repr(C)]
struct AttnP {
    n_q: u32,
    n_kv: u32,
    hd: u32,
    causal: u32,
    scale: f32,
    tokens: u32,
}

#[repr(C)]
struct SoftmaxP {
    cols: u32,
    ld: u32,
    q0: u32,
    causal: u32,
    scale: f32,
}

/// A strided matrix view into a buffer, for MPS.
struct View<'a> {
    buf: &'a Buf,
    offset: usize,
    row_bytes: usize,
    dtype: u32,
}

impl<'a> View<'a> {
    fn f32(buf: &'a Buf, offset: usize, row_bytes: usize) -> View<'a> {
        View {
            buf,
            offset,
            row_bytes,
            dtype: 0x1000_0000 | 32,
        }
    }
    fn f16(buf: &'a Buf, offset: usize, row_bytes: usize) -> View<'a> {
        View {
            buf,
            offset,
            row_bytes,
            dtype: 0x1000_0000 | 16,
        }
    }
}

fn bytes_of<T>(v: &T) -> &[u8] {
    unsafe { std::slice::from_raw_parts(v as *const T as *const u8, std::mem::size_of::<T>()) }
}

impl Metal {
    pub fn new() -> Result<Metal, String> {
        let _p = AutoreleasePool::new();
        let dev = unsafe { objc::MTLCreateSystemDefaultDevice() };
        if dev.is_null() {
            return Err("no Metal device".into());
        }
        let device_name = objc::to_string(msg!(Id; dev, c"name"));
        let queue = msg!(Id; dev, c"newCommandQueue");
        let src = objc::nsstring(SOURCE);
        let opts = msg!(Id; msg!(Id; objc::class(c"MTLCompileOptions"), c"alloc"), c"init");
        // exact transcendental functions: correctness over a few percent of speed
        msg!((); opts, c"setFastMathEnabled:", std::env::var("INBED_FAST_MATH").is_ok_and(|v| v == "1") => bool);
        let mut err: Id = std::ptr::null_mut();
        let lib = msg!(Id; dev, c"newLibraryWithSource:options:error:", src => Id, opts => Id, &mut err as *mut Id => *mut Id);
        objc::release(src);
        objc::release(opts);
        if lib.is_null() {
            return Err(format!(
                "Metal shader compile failed: {}",
                objc::error_string(err)
            ));
        }
        let mut psos = HashMap::new();
        for &k in KERNELS {
            let name = objc::nsstring(k);
            let f = msg!(Id; lib, c"newFunctionWithName:", name => Id);
            objc::release(name);
            if f.is_null() {
                return Err(format!("kernel {k} missing"));
            }
            let mut err: Id = std::ptr::null_mut();
            let pso = msg!(Id; dev, c"newComputePipelineStateWithFunction:error:", f => Id, &mut err as *mut Id => *mut Id);
            objc::release(f);
            if pso.is_null() {
                return Err(format!("pipeline {k}: {}", objc::error_string(err)));
            }
            psos.insert(k, pso);
        }
        objc::release(lib);
        let pool = Rc::new(RefCell::new(Pool::default()));
        let mut m = Metal {
            dev,
            queue,
            psos,
            pool,
            dummy: Buf {
                id: std::ptr::null_mut(),
                ptr: std::ptr::null_mut(),
                bytes: 0,
                pool: None,
            },
            pass: None,
            tokens: 0,
            device_name,
            flash: std::env::var("INBED_ATTN")
                .map(|v| v != "simple")
                .unwrap_or(true),
            timing: std::env::var("INBED_TIMING").is_ok_and(|v| v != "0"),
            pass_started: None,
            use_mps: std::env::var("INBED_GEMM")
                .map(|v| v != "native")
                .unwrap_or(true),
            mps: RefCell::new(HashMap::new()),
            starts: vec![],
            long_attn: crate::backend::long_seq(),
            profile: std::env::var("INBED_PROFILE")
                .ok()
                .filter(|v| v != "0")
                .map(|_| RefCell::new(HashMap::new())),
        };
        m.dummy = m.new_buf(256)?;
        Ok(m)
    }

    pub fn device_name(&self) -> &str {
        &self.device_name
    }

    fn new_buf(&self, bytes: usize) -> Result<Buf, String> {
        let id = msg!(Id; self.dev, c"newBufferWithLength:options:", bytes.max(16) => usize, 0usize => usize);
        if id.is_null() {
            return Err(format!("Metal: could not allocate {bytes} bytes"));
        }
        let ptr = msg!(*mut c_void; id, c"contents") as *mut u8;
        Ok(Buf {
            id,
            ptr,
            bytes,
            pool: None,
        })
    }

    /// Pooled scratch buffer (size rounded up to a power of two).
    fn scratch(&self, bytes: usize) -> Buf {
        let cls = bytes.max(4096).next_power_of_two();
        if let Some((id, ptr)) = self
            .pool
            .borrow_mut()
            .free
            .get_mut(&cls)
            .and_then(|v| v.pop())
        {
            return Buf {
                id,
                ptr,
                bytes: cls,
                pool: Some(self.pool.clone()),
            };
        }
        let mut b = self.new_buf(cls).expect("Metal allocation failed");
        self.pool.borrow_mut().allocated += cls;
        b.pool = Some(self.pool.clone());
        b
    }

    pub fn scratch_bytes(&self) -> usize {
        self.pool.borrow().allocated
    }

    /// Activations are allocated padded to a multiple of 64 rows so tiled kernels can read and
    /// write whole tiles; padding rows hold garbage and are never consumed.
    fn tensor(&self, rows: usize, cols: usize) -> MetalT {
        MetalT {
            buf: self.scratch(rows.next_multiple_of(64) * cols * 4),
            rows,
            cols,
        }
    }

    fn upload_u32(&self, v: &[u32]) -> Buf {
        let b = self.scratch(v.len().next_multiple_of(64) * 4);
        unsafe { std::ptr::copy_nonoverlapping(v.as_ptr() as *const u8, b.ptr, v.len() * 4) };
        b
    }

    fn enc(&self) -> Id {
        self.pass.as_ref().expect("begin() not called").enc.get()
    }

    fn dispatch(
        &self,
        kernel: &'static str,
        bufs: &[&Buf],
        params: &[u8],
        groups: MTLSize,
        threads: MTLSize,
    ) {
        let enc = self.enc();
        msg!((); enc, c"setComputePipelineState:", self.psos[kernel] => Id);
        for (i, b) in bufs.iter().enumerate() {
            msg!((); enc, c"setBuffer:offset:atIndex:", b.id => Id, 0usize => usize, i => usize);
        }
        if !params.is_empty() {
            msg!((); enc, c"setBytes:length:atIndex:", params.as_ptr() as *const c_void => *const c_void, params.len() => usize, bufs.len() => usize);
        }
        msg!((); enc, c"dispatchThreadgroups:threadsPerThreadgroup:", groups => MTLSize, threads => MTLSize);
        if let Some(prof) = &self.profile {
            let pass = self.pass();
            msg!((); pass.enc.get(), c"endEncoding");
            msg!((); pass.cb.get(), c"commit");
            msg!((); pass.cb.get(), c"waitUntilCompleted");
            let t =
                msg!(f64; pass.cb.get(), c"GPUEndTime") - msg!(f64; pass.cb.get(), c"GPUStartTime");
            let mut p = prof.borrow_mut();
            let e = p.entry(kernel).or_insert((0, 0.0));
            e.0 += 1;
            e.1 += t;
            pass.cb.set(msg!(Id; self.queue, c"commandBuffer"));
            pass.enc
                .set(msg!(Id; pass.cb.get(), c"computeCommandEncoder"));
        }
    }

    /// Encode `C = A · Wᵀ` with MPS between compute encoders (f32 activations, f16 weights).
    fn mps_matmul(&self, a: &Buf, w: &Buf, c: &Buf, m: usize, n: usize, k: usize) {
        self.mps_gemm(
            View::f32(a, 0, k * 4),
            View::f16(w, 0, k * 2),
            View::f32(c, 0, n * 4),
            m,
            n,
            k,
            true,
        );
    }

    /// `C[m×n] = A[m×k] · B` with `B` given as `[n×k]` (transpose_b) or `[k×n]`, over strided views.
    #[allow(clippy::too_many_arguments)]
    fn mps_gemm(&self, a: View, b: View, c: View, m: usize, n: usize, k: usize, transpose_b: bool) {
        let mut cache = self.mps.borrow_mut();
        if cache.len() > 512 {
            // in-flight kernels are retained by the command buffer
            for k in cache.values() {
                objc::release(*k);
            }
            cache.clear();
        }
        let dev = self.dev;
        let kern = *cache.entry((m, n, k, transpose_b)).or_insert_with(|| {
            let mm = msg!(Id; objc::class(c"MPSMatrixMultiplication"), c"alloc");
            msg!(Id; mm, c"initWithDevice:transposeLeft:transposeRight:resultRows:resultColumns:interiorColumns:alpha:beta:",
                dev => Id, false => bool, transpose_b => bool, m => usize, n => usize, k => usize, 1.0f64 => f64, 0.0f64 => f64)
        });
        drop(cache);
        let mat = |v: &View, rows: usize, cols: usize| -> Id {
            let desc = msg!(Id; objc::class(c"MPSMatrixDescriptor"), c"matrixDescriptorWithRows:columns:rowBytes:dataType:", rows => usize, cols => usize, v.row_bytes => usize, v.dtype => u32);
            let mx = msg!(Id; objc::class(c"MPSMatrix"), c"alloc");
            let mx = msg!(Id; mx, c"initWithBuffer:offset:descriptor:", v.buf.id => Id, v.offset => usize, desc => Id);
            msg!(Id; mx, c"autorelease")
        };
        let (rb, cb_) = if transpose_b { (n, k) } else { (k, n) };
        let (ma, mb, mc) = (mat(&a, m, k), mat(&b, rb, cb_), mat(&c, m, n));
        let pass = self.pass();
        msg!((); pass.enc.get(), c"endEncoding");
        msg!((); kern, c"encodeToCommandBuffer:leftMatrix:rightMatrix:resultMatrix:", pass.cb.get() => Id, ma => Id, mb => Id, mc => Id);
        if let Some(prof) = &self.profile {
            let cb = pass.cb.get();
            msg!((); cb, c"commit");
            msg!((); cb, c"waitUntilCompleted");
            let t = msg!(f64; cb, c"GPUEndTime") - msg!(f64; cb, c"GPUStartTime");
            let mut p = prof.borrow_mut();
            let e = p.entry("mps_gemm").or_insert((0, 0.0));
            e.0 += 1;
            e.1 += t;
            pass.cb.set(msg!(Id; self.queue, c"commandBuffer"));
        }
        pass.enc
            .set(msg!(Id; pass.cb.get(), c"computeCommandEncoder"));
    }

    /// Long sequences: per sequence, head and query chunk, S = Q·Kᵀ and O = P·V as MPS matmuls
    /// with a row-softmax kernel in between (causal chunks only multiply the keys they can see).
    fn attention_matmul(&self, qkv: &MetalT, out: &MetalT, s: &AttnShape, p: &AttnP) {
        const CHUNK: usize = 1024;
        let hd = s.head_dim;
        let stride = qkv.cols;
        let ocols = out.cols;
        let group = s.n_q / s.n_kv;
        let starts = self.starts.clone();
        let longest = starts
            .windows(2)
            .map(|w| (w[1] - w[0]) as usize)
            .max()
            .unwrap_or(0);
        let sbuf = self.scratch(CHUNK.min(longest) * longest.next_multiple_of(4) * 4);
        for w in starts.windows(2) {
            let (start, len) = (w[0] as usize, (w[1] - w[0]) as usize);
            for h in 0..s.n_q {
                let kvh = h / group;
                for q0 in (0..len).step_by(CHUNK) {
                    let c = CHUNK.min(len - q0);
                    let lk = if s.causal { q0 + c } else { len };
                    let ld = lk.next_multiple_of(4);
                    let q = View::f32(&qkv.buf, ((start + q0) * stride + h * hd) * 4, stride * 4);
                    let k = View::f32(
                        &qkv.buf,
                        (start * stride + (s.n_q + kvh) * hd) * 4,
                        stride * 4,
                    );
                    let v = View::f32(
                        &qkv.buf,
                        (start * stride + (s.n_q + s.n_kv + kvh) * hd) * 4,
                        stride * 4,
                    );
                    let sv = View::f32(&sbuf, 0, ld * 4);
                    self.mps_gemm(q, k, View::f32(&sbuf, 0, ld * 4), c, lk, hd, true);
                    let sp = SoftmaxP {
                        cols: lk as u32,
                        ld: ld as u32,
                        q0: q0 as u32,
                        causal: p.causal,
                        scale: p.scale,
                    };
                    self.dispatch(
                        "softmax_rows",
                        &[&sbuf],
                        bytes_of(&sp),
                        size(c, 1, 1),
                        size(256, 1, 1),
                    );
                    let o = View::f32(&out.buf, ((start + q0) * ocols + h * hd) * 4, ocols * 4);
                    self.mps_gemm(sv, v, o, c, hd, lk, false);
                }
            }
        }
    }

    /// Per-kernel (calls, GPU seconds) since the last call; empty unless INBED_PROFILE is set.
    pub fn take_profile(&self) -> Vec<(&'static str, u64, f64)> {
        let Some(p) = &self.profile else {
            return vec![];
        };
        let mut v: Vec<_> = p
            .borrow_mut()
            .drain()
            .map(|(k, (n, t))| (k, n, t))
            .collect();
        v.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap());
        v
    }

    fn pass(&self) -> &Pass {
        self.pass.as_ref().expect("begin() not called")
    }
}

impl Backend for Metal {
    type W = MetalW;
    type T = MetalT;

    fn name(&self) -> &'static str {
        "metal"
    }

    fn upload(&mut self, data: &[f32], rows: usize, cols: usize, kind: WeightKind) -> MetalW {
        assert_eq!(data.len(), rows * cols);
        let buf = match kind {
            WeightKind::Matrix | WeightKind::Table => {
                let b = self
                    .new_buf(data.len() * 2)
                    .expect("Metal allocation failed");
                let dst = unsafe { std::slice::from_raw_parts_mut(b.ptr as *mut u16, data.len()) };
                for (d, s) in dst.iter_mut().zip(data) {
                    *d = f32_to_f16(*s);
                }
                b
            }
            WeightKind::Vector => {
                let b = self
                    .new_buf(data.len() * 4)
                    .expect("Metal allocation failed");
                unsafe {
                    std::ptr::copy_nonoverlapping(data.as_ptr() as *const u8, b.ptr, data.len() * 4)
                };
                b
            }
        };
        MetalW { buf, rows, cols }
    }

    fn begin(&mut self, batch: &Batch) {
        self.pass = None;
        self.pass_started = Some(std::time::Instant::now());
        let pool = AutoreleasePool::new();
        let cb = msg!(Id; self.queue, c"commandBuffer");
        let enc = msg!(Id; cb, c"computeCommandEncoder");
        self.tokens = batch.tokens();
        self.starts = batch.starts.clone();
        self.pass = Some(Pass {
            _pool: pool,
            cb: cb.into(),
            enc: enc.into(),
            ids: self.upload_u32(&batch.ids),
            pos: self.upload_u32(&batch.positions),
            seq_end: self.upload_u32(&batch.seq_end),
        });
    }

    fn embed(&mut self, word: &MetalW, pos: Option<&MetalW>, type0: Option<&MetalW>) -> MetalT {
        let c = word.cols;
        let out = self.tensor(self.tokens, c);
        let p = EmbedP {
            cols: c as u32,
            has_pos: pos.is_some() as u32,
            has_type: type0.is_some() as u32,
        };
        let pass = self.pass();
        self.dispatch(
            "embed",
            &[
                &pass.ids,
                &pass.pos,
                &word.buf,
                pos.map(|w| &w.buf).unwrap_or(&self.dummy),
                type0.map(|w| &w.buf).unwrap_or(&self.dummy),
                &out.buf,
            ],
            bytes_of(&p),
            size(c.div_ceil(64), self.tokens, 1),
            size(64, 1, 1),
        );
        out
    }

    fn linear(
        &mut self,
        x: &MetalT,
        w: &MetalW,
        b: Option<&MetalW>,
        act: Act,
        residual: Option<&MetalT>,
    ) -> MetalT {
        assert_eq!(x.cols, w.cols, "linear: in features");
        let (m, n, k) = (x.rows, w.rows, w.cols);
        let out = self.tensor(m, n);
        let p = GemmP {
            m: m as u32,
            n: n as u32,
            k: k as u32,
            has_bias: b.is_some() as u32,
            has_res: residual.is_some() as u32,
            act: (act == Act::Gelu) as u32,
        };
        let bias = b.map(|b| &b.buf).unwrap_or(&self.dummy);
        let res = residual.map(|r| &r.buf).unwrap_or(&self.dummy);
        if self.use_mps {
            self.mps_matmul(&x.buf, &w.buf, &out.buf, m, n, k);
            if b.is_some() || residual.is_some() || act != Act::None {
                let bias = b.map(|b| &b.buf).unwrap_or(&self.dummy);
                self.dispatch(
                    "epilogue",
                    &[&out.buf, bias, res],
                    bytes_of(&p),
                    size(n.div_ceil(256), m, 1),
                    size(256, 1, 1),
                );
            }
        } else {
            self.dispatch(
                "gemm",
                &[&x.buf, &w.buf, bias, res, &out.buf],
                bytes_of(&p),
                size(n.div_ceil(64), m.div_ceil(64), 1),
                size(128, 1, 1),
            );
        }
        out
    }

    fn layer_norm(&mut self, x: &MetalT, g: &MetalW, b: &MetalW, eps: f32) -> MetalT {
        let out = self.tensor(x.rows, x.cols);
        let p = NormP {
            cols: x.cols as u32,
            eps,
        };
        self.dispatch(
            "layer_norm",
            &[&x.buf, &g.buf, &b.buf, &out.buf],
            bytes_of(&p),
            size(x.rows, 1, 1),
            size(256, 1, 1),
        );
        out
    }

    fn rms_norm(&mut self, x: &MetalT, g: &MetalW, eps: f32) -> MetalT {
        let out = self.tensor(x.rows, x.cols);
        let p = NormP {
            cols: x.cols as u32,
            eps,
        };
        self.dispatch(
            "rms_norm",
            &[&x.buf, &g.buf, &out.buf],
            bytes_of(&p),
            size(x.rows, 1, 1),
            size(256, 1, 1),
        );
        out
    }

    fn qk_norm_rope(
        &mut self,
        qkv: &MetalT,
        s: &AttnShape,
        qn: &MetalW,
        kn: &MetalW,
        eps: f32,
        theta: f32,
    ) {
        assert!(
            s.head_dim.is_multiple_of(64) && s.head_dim <= 256,
            "qk_norm_rope: head_dim must be a multiple of 64, ≤ 256"
        );
        let p = RopeP {
            stride: qkv.cols as u32,
            n_q: s.n_q as u32,
            n_kv: s.n_kv as u32,
            hd: s.head_dim as u32,
            eps,
            theta,
        };
        let pass = self.pass();
        self.dispatch(
            "qk_norm_rope",
            &[&qkv.buf, &pass.pos, &qn.buf, &kn.buf],
            bytes_of(&p),
            size(s.n_q + s.n_kv, qkv.rows, 1),
            size(32, 1, 1),
        );
    }

    fn attention(&mut self, qkv: &MetalT, s: &AttnShape) -> MetalT {
        assert!(
            s.head_dim.is_multiple_of(32) && s.head_dim <= 256,
            "attention: head_dim must be a multiple of 32, ≤ 256"
        );
        assert_eq!(qkv.cols, s.qkv_cols());
        let out = self.tensor(qkv.rows, s.n_q * s.head_dim);
        let p = AttnP {
            n_q: s.n_q as u32,
            n_kv: s.n_kv as u32,
            hd: s.head_dim as u32,
            causal: s.causal as u32,
            scale: 1.0 / (s.head_dim as f32).sqrt(),
            tokens: qkv.rows as u32,
        };
        let pass = self.pass();
        let flash = match s.head_dim {
            32 => Some("flash_attention_32"),
            64 => Some("flash_attention_64"),
            128 => Some("flash_attention_128"),
            _ => None,
        };
        let shortest = self
            .starts
            .windows(2)
            .map(|w| (w[1] - w[0]) as usize)
            .min()
            .unwrap_or(0);
        if self.use_mps && shortest >= self.long_attn {
            self.attention_matmul(qkv, &out, s, &p);
        } else if let (true, Some(kernel)) = (self.flash, flash) {
            self.dispatch(
                kernel,
                &[&qkv.buf, &pass.pos, &pass.seq_end, &out.buf],
                bytes_of(&p),
                size(qkv.rows.div_ceil(32), s.n_q, 1),
                size(128, 1, 1),
            );
        } else {
            self.dispatch(
                "attention",
                &[&qkv.buf, &pass.pos, &pass.seq_end, &out.buf],
                bytes_of(&p),
                size(s.n_q, qkv.rows, 1),
                size(32, 1, 1),
            );
        }
        out
    }

    fn silu_mul(&mut self, x: &MetalT) -> MetalT {
        let i = x.cols / 2;
        let out = self.tensor(x.rows, i);
        let ii = i as u32;
        self.dispatch(
            "silu_mul",
            &[&x.buf, &out.buf],
            bytes_of(&ii),
            size(i.div_ceil(256), x.rows, 1),
            size(256, 1, 1),
        );
        out
    }

    fn finish<R>(&mut self, x: &MetalT, f: impl FnOnce(&[f32], usize) -> R) -> R {
        let pass = self.pass.take().expect("begin() not called");
        let cb = pass.cb.get();
        msg!((); pass.enc.get(), c"endEncoding");
        let t_commit = std::time::Instant::now();
        msg!((); cb, c"commit");
        msg!((); cb, c"waitUntilCompleted");
        let status = msg!(usize; cb, c"status");
        if status != 4 {
            let e = objc::error_string(msg!(Id; cb, c"error"));
            panic!("Metal command buffer failed (status {status}): {e}");
        }
        if self.timing {
            let gpu = msg!(f64; cb, c"GPUEndTime") - msg!(f64; cb, c"GPUStartTime");
            let encode = self
                .pass_started
                .map(|t| (t_commit - t).as_secs_f64())
                .unwrap_or(0.0);
            eprintln!(
                "metal pass: {} tokens, encode {:.2} ms, gpu {:.2} ms, wall {:.2} ms",
                x.rows,
                encode * 1e3,
                gpu * 1e3,
                self.pass_started
                    .map(|t| t.elapsed().as_secs_f64() * 1e3)
                    .unwrap_or(0.0)
            );
        }
        let data = unsafe { std::slice::from_raw_parts(x.buf.ptr as *const f32, x.rows * x.cols) };
        let r = f(data, x.cols);
        drop(pass);
        r
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cpu::Cpu;

    fn rnd(n: usize, seed: u64, scale: f32) -> Vec<f32> {
        let mut s = seed;
        (0..n)
            .map(|_| {
                s = s
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                (((s >> 33) as f32 / (1u64 << 31) as f32) * 2.0 - 1.0) * scale
            })
            .collect()
    }

    /// values exact in f16, so both backends see identical weights
    fn h(v: Vec<f32>) -> Vec<f32> {
        v.into_iter()
            .map(|x| crate::safetensors::f16_to_f32(f32_to_f16(x)))
            .collect()
    }

    fn close(a: &[f32], b: &[f32], tol: f32, what: &str) {
        assert_eq!(a.len(), b.len(), "{what}: len");
        let worst = a
            .iter()
            .zip(b)
            .map(|(x, y)| (x - y).abs() / (1.0 + y.abs()))
            .fold(0f32, f32::max);
        eprintln!("{what}: max rel err {worst:.2e}");
        assert!(worst <= tol, "{what}: max rel err {worst} > {tol}");
    }

    fn batch() -> Batch {
        let seqs: Vec<Vec<u32>> = vec![
            (0..37).collect(),
            (0..5).collect(),
            (0..70).map(|i| i * 2).collect(),
            vec![3],
        ];
        let refs: Vec<&[u32]> = seqs.iter().map(|s| s.as_slice()).collect();
        Batch::new(&refs)
    }

    fn gemm_case<B: Backend>(
        b: &mut B,
        n: usize,
        k: usize,
        act: Act,
        bias: bool,
        res: bool,
    ) -> Vec<f32> {
        let bt = batch();
        let table = b.upload(&h(rnd(200 * k, 1, 1.0)), 200, k, WeightKind::Table);
        let rtab = b.upload(&h(rnd(200 * n, 2, 1.0)), 200, n, WeightKind::Table);
        let w = b.upload(&h(rnd(n * k, 3, 0.2)), n, k, WeightKind::Matrix);
        let bv = b.upload(&rnd(n, 4, 1.0), 1, n, WeightKind::Vector);
        b.begin(&bt);
        let x = b.embed(&table, Some(&table), None);
        let r = b.embed(&rtab, None, None);
        let y = b.linear(&x, &w, bias.then_some(&bv), act, res.then_some(&r));
        b.finish(&y, |d, _| d.to_vec())
    }

    fn block_case<B: Backend>(b: &mut B, causal: bool) -> Vec<f32> {
        block_case_on(b, causal, &batch())
    }

    fn block_case_on<B: Backend>(b: &mut B, causal: bool, bt: &Batch) -> Vec<f32> {
        let (hd, hid) = (64, 128);
        let s = AttnShape {
            n_q: 4,
            n_kv: 2,
            head_dim: hd,
            causal,
        };
        let wt = b.upload(&h(rnd(2048 * hid, 11, 1.0)), 2048, hid, WeightKind::Table);
        let wqkv = b.upload(
            &h(rnd(s.qkv_cols() * hid, 12, 0.2)),
            s.qkv_cols(),
            hid,
            WeightKind::Matrix,
        );
        let g = b.upload(
            &rnd(hid, 13, 0.1)
                .iter()
                .map(|x| 1.0 + x)
                .collect::<Vec<_>>(),
            1,
            hid,
            WeightKind::Vector,
        );
        let bb = b.upload(&rnd(hid, 14, 1.0), 1, hid, WeightKind::Vector);
        let qn = b.upload(
            &rnd(hd, 15, 0.5).iter().map(|x| 1.0 + x).collect::<Vec<_>>(),
            1,
            hd,
            WeightKind::Vector,
        );
        let kn = b.upload(
            &rnd(hd, 16, 0.5).iter().map(|x| 1.0 + x).collect::<Vec<_>>(),
            1,
            hd,
            WeightKind::Vector,
        );
        let wo = b.upload(
            &h(rnd(2 * hid * s.n_q * hd, 17, 0.1)),
            2 * hid,
            s.n_q * hd,
            WeightKind::Matrix,
        );
        b.begin(bt);
        let e = b.embed(&wt, Some(&wt), Some(&wt));
        let x = b.layer_norm(&e, &g, &bb, 1e-12);
        let x = b.rms_norm(&x, &g, 1e-6);
        let qkv = b.linear(&x, &wqkv, None, Act::None, None);
        b.qk_norm_rope(&qkv, &s, &qn, &kn, 1e-6, 1e6);
        let a = b.attention(&qkv, &s);
        let o = b.linear(&a, &wo, None, Act::None, Some(&x).filter(|_| false));
        let m = b.silu_mul(&o);
        b.finish(&m, |d, _| d.to_vec())
    }

    #[test]
    fn kernels_match_cpu() {
        let mut gpu = match Metal::new() {
            Ok(g) => g,
            Err(e) if e == "no Metal device" => {
                eprintln!("skip: {e}");
                return;
            }
            Err(e) => panic!("{e}"),
        };
        let mut cpu = Cpu::new();
        let mut native = Metal::new().unwrap();
        native.use_mps = false;
        for (n, k, act, bias, res) in [
            (100, 64, Act::None, false, false),
            (64, 96, Act::Gelu, true, false),
            (129, 32, Act::None, true, true),
            (384, 384, Act::Gelu, true, true),
        ] {
            let c = gemm_case(&mut cpu, n, k, act, bias, res);
            let g = gemm_case(&mut gpu, n, k, act, bias, res);
            close(
                &g,
                &c,
                1e-4,
                &format!("gemm(mps) n={n} k={k} {act:?} bias={bias} res={res}"),
            );
            let g2 = gemm_case(&mut native, n, k, act, bias, res);
            close(
                &g2,
                &c,
                1e-4,
                &format!("gemm(native) n={n} k={k} {act:?} bias={bias} res={res}"),
            );
        }
        for causal in [false, true] {
            let c = block_case(&mut cpu, causal);
            let g = block_case(&mut gpu, causal);
            close(&g, &c, 1e-3, &format!("block(flash) causal={causal}"));
            gpu.flash = false;
            let g = block_case(&mut gpu, causal);
            gpu.flash = true;
            close(&g, &c, 1e-3, &format!("block(simple) causal={causal}"));
        }
        // long sequences: matmul attention
        let a: Vec<u32> = (0..1100).map(|i| i % 200).collect();
        let b2: Vec<u32> = (0..1300).map(|i| (i * 7) % 200).collect();
        let long = Batch::new(&[&a, &b2]);
        for causal in [false, true] {
            gpu.long_attn = 1024;
            let g = block_case_on(&mut gpu, causal, &long);
            let c = block_case_on(&mut cpu, causal, &long);
            close(
                &g,
                &c,
                1e-3,
                &format!("block(matmul attention) causal={causal}"),
            );
            gpu.long_attn = usize::MAX;
            let f = block_case_on(&mut gpu, causal, &long);
            close(&f, &c, 1e-3, &format!("block(flash, long) causal={causal}"));
        }
        gpu.long_attn = crate::backend::long_seq();
    }

    /// cargo test --release --lib gemm_speed -- --ignored --nocapture
    #[test]
    #[ignore]
    fn gemm_speed() {
        let mut gpu = Metal::new().unwrap();
        for (m, n, k) in [
            (27, 4096, 1024),
            (4096, 4096, 1024),
            (4096, 6144, 1024),
            (4096, 1024, 3072),
            (4096, 384, 384),
        ] {
            let ids: Vec<u32> = (0..m as u32).map(|i| i % 200).collect();
            let bt = Batch::new(&[&ids]);
            let table = gpu.upload(&rnd(200 * k, 1, 1.0), 200, k, WeightKind::Table);
            let w = gpu.upload(&rnd(n * k, 3, 0.2), n, k, WeightKind::Matrix);
            let reps = if m < 100 { 200 } else { 20 };
            gpu.begin(&bt);
            let x = gpu.embed(&table, None, None);
            gpu.finish(&x, |_, _| ());
            let t = std::time::Instant::now();
            gpu.begin(&bt);
            let x = gpu.embed(&table, None, None);
            let mut last = None;
            for _ in 0..reps {
                last = Some(gpu.linear(&x, &w, None, Act::None, None));
            }
            gpu.finish(last.as_ref().unwrap(), |_, _| ());
            let s = t.elapsed().as_secs_f64() / reps as f64;
            eprintln!(
                "gemm {m}x{n}x{k}: {:.3} ms  {:.2} TFLOPS  {:.0} GB/s(weights)",
                s * 1e3,
                2.0 * (m * n * k) as f64 / s / 1e12,
                (n * k * 2) as f64 / s / 1e9
            );
        }
    }
}

#[cfg(test)]
mod mps_probe {
    use super::*;
    #[link(name = "MetalPerformanceShaders", kind = "framework")]
    unsafe extern "C" {}

    fn mat(buf: Id, rows: usize, cols: usize, dtype: u32) -> Id {
        let desc = msg!(Id; objc::class(c"MPSMatrixDescriptor"), c"matrixDescriptorWithRows:columns:rowBytes:dataType:", rows => usize, cols => usize, cols * (dtype as usize & 0xffff) / 8 => usize, dtype => u32);
        let m = msg!(Id; objc::class(c"MPSMatrix"), c"alloc");
        msg!(Id; m, c"initWithBuffer:descriptor:", buf => Id, desc => Id)
    }

    /// cargo test --release --lib mps_speed -- --ignored --nocapture
    #[test]
    #[ignore]
    fn mps_speed() {
        let gpu = Metal::new().unwrap();
        let _p = AutoreleasePool::new();
        for (wdt, name) in [(0x1000_0000u32 | 32, "f32"), (0x1000_0000u32 | 16, "f16")] {
            for (m, n, k) in [
                (27usize, 4096usize, 1024usize),
                (4096, 4096, 1024),
                (4096, 1024, 3072),
            ] {
                let a = gpu.new_buf(m * k * 4).unwrap();
                let wsz = (wdt & 0xffff) as usize / 8;
                let b = gpu.new_buf(n * k * wsz).unwrap();
                let c = gpu.new_buf(m * n * 4).unwrap();
                let (ma, mb, mc) = (
                    mat(a.id, m, k, 0x1000_0000 | 32),
                    mat(b.id, n, k, wdt),
                    mat(c.id, m, n, 0x1000_0000 | 32),
                );
                let mm = msg!(Id; objc::class(c"MPSMatrixMultiplication"), c"alloc");
                let mm = msg!(Id; mm, c"initWithDevice:transposeLeft:transposeRight:resultRows:resultColumns:interiorColumns:alpha:beta:",
                    gpu.dev => Id, false => bool, true => bool, m => usize, n => usize, k => usize, 1.0f64 => f64, 0.0f64 => f64);
                let reps = 20;
                let cb = msg!(Id; gpu.queue, c"commandBuffer");
                for _ in 0..reps {
                    msg!((); mm, c"encodeToCommandBuffer:leftMatrix:rightMatrix:resultMatrix:", cb => Id, ma => Id, mb => Id, mc => Id);
                }
                let t = std::time::Instant::now();
                msg!((); cb, c"commit");
                msg!((); cb, c"waitUntilCompleted");
                let s = t.elapsed().as_secs_f64() / reps as f64;
                let st = msg!(usize; cb, c"status");
                eprintln!(
                    "MPS A=f32 W={name} {m}x{n}x{k}: {:.3} ms {:.2} TFLOPS (status {st})",
                    s * 1e3,
                    2.0 * (m * n * k) as f64 / s / 1e12
                );
            }
        }
    }
}
