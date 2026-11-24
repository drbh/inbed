//! CUDA backend: f16 weights, f32 activations, one stream per backend, kernels compiled with
//! NVRTC at startup for the device's architecture, matmuls on the tensor cores through cuBLASLt.
//!
//! Everything is loaded with `dlopen` (see `ffi.rs`): no crates, and no link-time CUDA.
//! Work is queued asynchronously on the stream; `finish` copies the result back and waits.
//!
//! A tensor holds f32 values, an f16 copy, or both. Matmuls read f16, so the kernels that feed
//! one (norms, SiLU, attention) write the f16 copy directly, and those whose output only a
//! matmul reads (SiLU, attention) write nothing else. Bias and residual adds are fused into the
//! matmul (`D = W·X + bias + C`).

mod ffi;

use crate::backend::{Act, AttnShape, Backend, Batch, WeightKind};
use crate::safetensors::f32_to_f16;
use ffi::*;
use std::cell::{Cell, RefCell};
use std::collections::HashMap;
use std::ffi::{CStr, CString, c_char, c_int, c_void};
use std::rc::Rc;

const SOURCE: &str = include_str!("kernels.cu");
const KERNELS: &[&str] = &[
    "embed",
    "to_f16",
    "to_f32",
    "epilogue",
    "rms_norm",
    "layer_norm",
    "rope_table",
    "qk_norm_rope",
    "silu_mul",
    "attention_32",
    "attention_64",
    "attention_128",
    "fattn_32",
    "fattn_64",
    "fattn_128",
];
/// queries per tile of the f32 attention kernel (QT in kernels.cu)
const QT: usize = 32;
/// keys per shared-memory tile of the f32 attention kernel (KT in kernels.cu)
const KT: usize = 32;
/// queries and keys per tile of the tensor-core attention kernel (FQ, FK in kernels.cu)
const FQ: usize = 64;
const FK: usize = 64;
/// cuBLASLt workspace
const WORKSPACE: usize = 32 << 20;

/// Whether a CUDA device is usable here: the libraries load and the driver reports a device.
pub fn available() -> bool {
    let Ok(l) = Libs::load() else { return false };
    let mut n = 0;
    unsafe { (l.cu.init)(0) == 0 && (l.cu.device_get_count)(&mut n) == 0 && n > 0 }
}

/// What every buffer needs to free itself: the driver, and the pool scratch buffers return to.
struct Inner {
    libs: Libs,
    dev: CuDevice,
    stream: CuStream,
    /// free scratch buffers by size class
    free: RefCell<HashMap<usize, Vec<CuDevPtr>>>,
    allocated: Cell<usize>,
}

impl Inner {
    fn check(&self, r: CuResult, what: &str) {
        if r != 0 {
            panic!("CUDA {what} failed: {}", self.libs.cu_error(r));
        }
    }

    fn alloc(&self, bytes: usize) -> Result<CuDevPtr, String> {
        let mut p: CuDevPtr = 0;
        let r = unsafe { (self.libs.cu.mem_alloc)(&mut p, bytes.max(256)) };
        if r != 0 {
            return Err(format!(
                "CUDA: could not allocate {} MiB: {}",
                bytes >> 20,
                self.libs.cu_error(r)
            ));
        }
        Ok(p)
    }
}

impl Drop for Inner {
    fn drop(&mut self) {
        unsafe {
            (self.libs.cu.stream_synchronize)(self.stream);
            for v in self.free.borrow().values() {
                for &p in v {
                    (self.libs.cu.mem_free)(p);
                }
            }
            (self.libs.cu.stream_destroy)(self.stream);
            (self.libs.cu.primary_ctx_release)(self.dev);
        }
    }
}

/// Device memory. Pooled (scratch) buffers go back to their size class on drop; since all work
/// is on one stream, a later kernel reusing one is ordered after every earlier use.
pub struct Buf {
    ptr: CuDevPtr,
    bytes: usize,
    pooled: bool,
    inner: Rc<Inner>,
}

impl Drop for Buf {
    fn drop(&mut self) {
        if self.pooled {
            self.inner
                .free
                .borrow_mut()
                .entry(self.bytes)
                .or_default()
                .push(self.ptr);
        } else {
            unsafe {
                (self.inner.libs.cu.stream_synchronize)(self.inner.stream);
                (self.inner.libs.cu.mem_free)(self.ptr);
            }
        }
    }
}

pub struct CudaW {
    buf: Buf,
    rows: usize,
    cols: usize,
}

pub struct CudaT {
    f32: Option<Buf>,
    f16: Option<Buf>,
    rows: usize,
    cols: usize,
}

impl CudaT {
    fn f32(&self) -> CuDevPtr {
        self.f32.as_ref().expect("tensor has no f32 values").ptr
    }
}

/// Page-locked host memory, for asynchronous copies.
struct Pinned {
    ptr: *mut u8,
    bytes: usize,
}

impl Pinned {
    fn empty() -> Pinned {
        Pinned {
            ptr: std::ptr::null_mut(),
            bytes: 0,
        }
    }
}

struct Pass {
    ids: Buf,
    pos: Buf,
    tiles: Buf,
    n_tiles: usize,
    longest: usize,
}

/// (m, n, k, bias, residual)
type PlanKey = (usize, usize, usize, bool, bool);

/// A cached cuBLASLt matmul: descriptor, layouts and the algorithm the heuristic picked.
struct Plan {
    desc: LtDesc,
    a: LtLayout,
    b: LtLayout,
    c: LtLayout,
    algo: LtAlgo,
}

/// RoPE cos/sin table for (theta, head_dim), covering positions < len.
struct RopeTable {
    theta: u32,
    hd: usize,
    len: usize,
    buf: Buf,
}

pub struct Cuda {
    inner: Rc<Inner>,
    module: CuModule,
    funcs: HashMap<&'static str, CuFunction>,
    lt: LtHandle,
    pref: LtPref,
    plans: RefCell<HashMap<PlanKey, Plan>>,
    workspace: Buf,
    dummy: Buf,
    pass: Option<Pass>,
    rope: Option<RopeTable>,
    /// tensor-core attention (default) or the f32 kernel (INBED_ATTN=simple)
    mma: bool,
    tokens: usize,
    device_name: String,
    arch: String,
    staging: Pinned,
    readback: Pinned,
    /// INBED_TIMING=1: print wall time per pass
    timing: bool,
    pass_started: Option<std::time::Instant>,
    /// INBED_PROFILE=1: synchronize around every kernel and add up its time
    profile: Option<RefCell<HashMap<&'static str, (u64, f64)>>>,
}

fn bytes_to_string(b: &[c_char]) -> String {
    unsafe { CStr::from_ptr(b.as_ptr()) }
        .to_string_lossy()
        .into_owned()
}

/// Pointers to kernel arguments, in order, for `cuLaunchKernel`.
macro_rules! args {
    ($($a:expr),* $(,)?) => {
        [$( &$a as *const _ as *mut c_void ),*]
    };
}

fn attn_smem(hd: usize) -> usize {
    (QT * hd + KT * (hd + 4) + KT * hd) * 4
}

fn fattn_smem(hd: usize) -> usize {
    (FQ * (hd + 8) + FK * (hd + 8) + hd * (FK + 8)) * 2
}

impl Cuda {
    pub fn new() -> Result<Cuda, String> {
        let libs = Libs::load()?;
        let cu = &libs.cu;
        let ck = |r: CuResult, what: &str| -> Result<(), String> {
            if r == 0 {
                Ok(())
            } else {
                Err(format!("CUDA {what}: {}", libs.cu_error(r)))
            }
        };
        let ordinal: c_int = std::env::var("INBED_CUDA_DEVICE")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(0);
        let (mut dev, mut ctx, mut stream) = (0, std::ptr::null_mut(), std::ptr::null_mut());
        let (mut major, mut minor) = (0, 0);
        let mut name = [0 as c_char; 256];
        unsafe {
            ck((cu.init)(0), "init")?;
            let mut n = 0;
            ck((cu.device_get_count)(&mut n), "device count")?;
            if n == 0 {
                return Err("no CUDA device".into());
            }
            ck((cu.device_get)(&mut dev, ordinal), "device")?;
            ck(
                (cu.device_get_name)(name.as_mut_ptr(), 256, dev),
                "device name",
            )?;
            ck(
                (cu.device_get_attribute)(
                    &mut major,
                    CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
                    dev,
                ),
                "compute capability",
            )?;
            ck(
                (cu.device_get_attribute)(
                    &mut minor,
                    CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,
                    dev,
                ),
                "compute capability",
            )?;
            ck((cu.primary_ctx_retain)(&mut ctx, dev), "context")?;
            ck((cu.ctx_set_current)(ctx), "set context")?;
            ck((cu.stream_create)(&mut stream, 1), "stream")?; // CU_STREAM_NON_BLOCKING
        }
        let arch = format!("sm_{major}{minor}");
        let cubin = compile(&libs, &arch)?;
        let mut module = std::ptr::null_mut();
        unsafe {
            ck(
                (cu.module_load_data)(&mut module, cubin.as_ptr() as *const c_void),
                "load module",
            )?
        };
        let mut funcs = HashMap::new();
        for &k in KERNELS {
            let cname = CString::new(k).unwrap();
            let mut f = std::ptr::null_mut();
            unsafe { ck((cu.module_get_function)(&mut f, module, cname.as_ptr()), k)? };
            let smem = match (k.strip_prefix("attention_"), k.strip_prefix("fattn_")) {
                (Some(hd), _) => Some(attn_smem(hd.parse().unwrap())),
                (_, Some(hd)) => Some(fattn_smem(hd.parse().unwrap())),
                _ => None,
            };
            if let Some(smem) = smem {
                let smem = smem as c_int;
                unsafe {
                    ck(
                        (cu.func_set_attribute)(
                            f,
                            CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                            smem,
                        ),
                        "attention shared memory",
                    )?
                };
            }
            funcs.insert(k, f);
        }
        let (mut lt, mut pref) = (std::ptr::null_mut(), std::ptr::null_mut());
        unsafe {
            if (libs.lt.create)(&mut lt) != 0 || (libs.lt.pref_create)(&mut pref) != 0 {
                return Err("cublasLtCreate failed".into());
            }
            let ws = WORKSPACE as u64;
            (libs.lt.pref_set)(
                pref,
                CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                &ws as *const u64 as *const c_void,
                8,
            );
        }
        let inner = Rc::new(Inner {
            libs,
            dev,
            stream,
            free: RefCell::new(HashMap::new()),
            allocated: Cell::new(0),
        });
        let raw = |bytes: usize| -> Result<Buf, String> {
            Ok(Buf {
                ptr: inner.alloc(bytes)?,
                bytes,
                pooled: false,
                inner: inner.clone(),
            })
        };
        Ok(Cuda {
            module,
            funcs,
            lt,
            pref,
            plans: RefCell::new(HashMap::new()),
            workspace: raw(WORKSPACE)?,
            dummy: raw(256)?,
            pass: None,
            rope: None,
            mma: std::env::var("INBED_ATTN")
                .map(|v| v != "simple")
                .unwrap_or(true),
            tokens: 0,
            device_name: bytes_to_string(&name),
            arch,
            staging: Pinned::empty(),
            readback: Pinned::empty(),
            timing: std::env::var("INBED_TIMING").is_ok_and(|v| v != "0"),
            pass_started: None,
            profile: std::env::var("INBED_PROFILE")
                .ok()
                .filter(|v| v != "0")
                .map(|_| RefCell::new(HashMap::new())),
            inner,
        })
    }

    pub fn device_name(&self) -> &str {
        &self.device_name
    }

    pub fn arch(&self) -> &str {
        &self.arch
    }

    /// Bytes held by the scratch pool.
    pub fn scratch_bytes(&self) -> usize {
        self.inner.allocated.get()
    }

    /// (free, total) device memory in bytes.
    pub fn mem_info(&self) -> (usize, usize) {
        let (mut f, mut t) = (0, 0);
        unsafe { (self.inner.libs.cu.mem_get_info)(&mut f, &mut t) };
        (f, t)
    }

    /// Per-kernel (calls, seconds) since the last call; empty unless INBED_PROFILE is set.
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

    /// Run `f` and, when profiling, charge its device time to `name`.
    fn timed(&self, name: &'static str, f: impl FnOnce()) {
        let Some(p) = &self.profile else {
            return f();
        };
        let sync = || unsafe { (self.cu().stream_synchronize)(self.inner.stream) };
        sync();
        let t = std::time::Instant::now();
        f();
        sync();
        let mut p = p.borrow_mut();
        let e = p.entry(name).or_insert((0, 0.0));
        e.0 += 1;
        e.1 += t.elapsed().as_secs_f64();
    }

    fn cu(&self) -> &Driver {
        &self.inner.libs.cu
    }

    /// Pooled scratch buffer, size rounded up to a power of two.
    fn scratch(&self, bytes: usize) -> Buf {
        let cls = bytes.max(4096).next_power_of_two();
        let ptr = self
            .inner
            .free
            .borrow_mut()
            .get_mut(&cls)
            .and_then(|v| v.pop());
        let ptr = match ptr {
            Some(p) => p,
            None => {
                let p = match self.inner.alloc(cls) {
                    Ok(p) => p,
                    Err(_) => {
                        // out of memory: release the pool's idle buffers and retry once
                        self.trim();
                        self.inner.alloc(cls).unwrap_or_else(|e| panic!("{e}"))
                    }
                };
                self.inner.allocated.set(self.inner.allocated.get() + cls);
                p
            }
        };
        Buf {
            ptr,
            bytes: cls,
            pooled: true,
            inner: self.inner.clone(),
        }
    }

    /// Free every idle scratch buffer.
    pub fn trim(&self) {
        unsafe { (self.cu().stream_synchronize)(self.inner.stream) };
        let mut free = self.inner.free.borrow_mut();
        for (cls, v) in free.drain() {
            for p in v {
                unsafe { (self.cu().mem_free)(p) };
                self.inner.allocated.set(self.inner.allocated.get() - cls);
            }
        }
    }

    fn tensor(&self, rows: usize, cols: usize, f32: bool, f16: bool) -> CudaT {
        CudaT {
            f32: f32.then(|| self.scratch(rows * cols * 4)),
            f16: f16.then(|| self.scratch(rows * cols * 2)),
            rows,
            cols,
        }
    }

    fn launch(
        &self,
        kernel: &'static str,
        grid: (usize, usize),
        block: usize,
        smem: usize,
        args: &mut [*mut c_void],
    ) {
        let f = self.funcs[kernel];
        self.timed(kernel, || {
            let r = unsafe {
                (self.cu().launch_kernel)(
                    f,
                    grid.0 as u32,
                    grid.1 as u32,
                    1,
                    block as u32,
                    1,
                    1,
                    smem as u32,
                    self.inner.stream,
                    args.as_mut_ptr(),
                    std::ptr::null_mut(),
                )
            };
            self.inner.check(r, kernel);
        });
    }

    fn convert(&self, kernel: &'static str, src: CuDevPtr, dst: CuDevPtr, n: usize) {
        let blocks = n.div_ceil(256).min(65535 * 4);
        self.launch(kernel, (blocks, 1), 256, 0, &mut args![src, dst, n]);
    }

    fn grow(&self, p: &mut Pinned, bytes: usize) {
        if p.bytes >= bytes {
            return;
        }
        unsafe {
            (self.cu().stream_synchronize)(self.inner.stream);
            if !p.ptr.is_null() {
                (self.cu().mem_free_host)(p.ptr as *mut c_void);
            }
            let bytes = bytes.next_power_of_two();
            let mut ptr = std::ptr::null_mut();
            self.inner.check(
                (self.cu().mem_alloc_host)(&mut ptr, bytes),
                "host allocation",
            );
            p.ptr = ptr as *mut u8;
            p.bytes = bytes;
        }
    }

    /// The cuBLASLt plan for `D[m×n] = X[m×k] · Wᵀ (+ bias) (+ C)`, row-major. In cuBLAS's
    /// column-major terms that is `Dᵀ[n×m] = op(W)·Xᵀ` with W stored as `[k×n]` (transposed),
    /// so the bias runs along D's rows, as cuBLASLt's bias epilogue expects.
    fn plan(
        &self,
        m: usize,
        n: usize,
        k: usize,
        bias: bool,
        res: bool,
    ) -> std::cell::RefMut<'_, Plan> {
        let key = (m, n, k, bias, res);
        let mut plans = self.plans.borrow_mut();
        if !plans.contains_key(&key) {
            if plans.len() > 1024 {
                for p in plans.values() {
                    self.destroy_plan(p);
                }
                plans.clear();
            }
            let lt = &self.inner.libs.lt;
            let mut p = Plan {
                desc: std::ptr::null_mut(),
                a: std::ptr::null_mut(),
                b: std::ptr::null_mut(),
                c: std::ptr::null_mut(),
                algo: LtAlgo { data: [0; 8] },
            };
            let set = |d: LtDesc, attr: c_int, v: &[u8]| unsafe {
                (lt.desc_set)(d, attr, v.as_ptr() as *const c_void, v.len());
            };
            unsafe {
                assert_eq!(
                    (lt.desc_create)(&mut p.desc, CUBLAS_COMPUTE_32F, CUDA_R_32F),
                    0,
                    "cublasLtMatmulDescCreate"
                );
                set(
                    p.desc,
                    CUBLASLT_MATMUL_DESC_TRANSA,
                    &CUBLAS_OP_T.to_le_bytes(),
                );
                set(
                    p.desc,
                    CUBLASLT_MATMUL_DESC_TRANSB,
                    &CUBLAS_OP_N.to_le_bytes(),
                );
                let epi = if bias {
                    CUBLASLT_EPILOGUE_BIAS
                } else {
                    CUBLASLT_EPILOGUE_DEFAULT
                };
                set(p.desc, CUBLASLT_MATMUL_DESC_EPILOGUE, &epi.to_le_bytes());
                if bias {
                    // the heuristic wants a pointer; the real one is set per call
                    set(
                        p.desc,
                        CUBLASLT_MATMUL_DESC_BIAS_POINTER,
                        &self.dummy.ptr.to_le_bytes(),
                    );
                }
                (lt.layout_create)(&mut p.a, CUDA_R_16F, k as u64, n as u64, k as i64);
                (lt.layout_create)(&mut p.b, CUDA_R_16F, k as u64, m as u64, k as i64);
                (lt.layout_create)(&mut p.c, CUDA_R_32F, n as u64, m as u64, n as i64);
                let mut h = [LtHeuristic {
                    algo: LtAlgo { data: [0; 8] },
                    workspace_size: 0,
                    state: 0,
                    waves_count: 0.0,
                    reserved: [0; 4],
                }];
                let mut found = 0;
                let r = (lt.heuristic)(
                    self.lt,
                    p.desc,
                    p.a,
                    p.b,
                    p.c,
                    p.c,
                    self.pref,
                    1,
                    h.as_mut_ptr(),
                    &mut found,
                );
                if r != 0 || found == 0 {
                    panic!(
                        "cuBLASLt: no algorithm for {m}x{n}x{k} (bias {bias}, residual {res}): status {r}"
                    );
                }
                p.algo = h[0].algo;
            }
            plans.insert(key, p);
        }
        std::cell::RefMut::map(plans, |p| p.get_mut(&key).unwrap())
    }

    fn destroy_plan(&self, p: &Plan) {
        let lt = &self.inner.libs.lt;
        unsafe {
            (lt.layout_destroy)(p.a);
            (lt.layout_destroy)(p.b);
            (lt.layout_destroy)(p.c);
            (lt.desc_destroy)(p.desc);
        }
    }

    /// `out = x · wᵀ (+ bias) (+ residual)` on the tensor cores, f16 inputs, f32 accumulation.
    fn gemm(
        &self,
        x16: CuDevPtr,
        w: &CudaW,
        out: CuDevPtr,
        m: usize,
        bias: Option<CuDevPtr>,
        res: Option<CuDevPtr>,
    ) {
        let (n, k) = (w.rows, w.cols);
        let plan = self.plan(m, n, k, bias.is_some(), res.is_some());
        let lt = &self.inner.libs.lt;
        if let Some(b) = bias {
            unsafe {
                (lt.desc_set)(
                    plan.desc,
                    CUBLASLT_MATMUL_DESC_BIAS_POINTER,
                    &b as *const u64 as *const c_void,
                    8,
                )
            };
        }
        let alpha = 1f32;
        let beta = if res.is_some() { 1f32 } else { 0f32 };
        let c = res.unwrap_or(out);
        self.timed("gemm", || {
            let r = unsafe {
                (lt.matmul)(
                    self.lt,
                    plan.desc,
                    &alpha as *const f32 as *const c_void,
                    w.buf.ptr as *const c_void,
                    plan.a,
                    x16 as *const c_void,
                    plan.b,
                    &beta as *const f32 as *const c_void,
                    c as *const c_void,
                    plan.c,
                    out as *mut c_void,
                    plan.c,
                    &plan.algo,
                    self.workspace.ptr as *mut c_void,
                    WORKSPACE,
                    self.inner.stream,
                )
            };
            if r != 0 {
                panic!("cublasLtMatmul {m}x{n}x{k} failed ({r})");
            }
        });
    }

    /// The RoPE table for this pass, computed when (theta, head_dim) change or a longer
    /// sequence arrives.
    fn rope_table(&mut self, hd: usize, theta: f32, longest: usize) -> CuDevPtr {
        let fresh = match &self.rope {
            Some(r) => r.theta != theta.to_bits() || r.hd != hd || r.len < longest,
            None => true,
        };
        if fresh {
            let len = longest.next_power_of_two().max(512);
            let n = len * hd / 2;
            let buf = self.scratch(n * 8);
            let (l, h) = (len as c_int, hd as c_int);
            self.launch(
                "rope_table",
                (n.div_ceil(256), 1),
                256,
                0,
                &mut args![buf.ptr, l, h, theta],
            );
            self.rope = Some(RopeTable {
                theta: theta.to_bits(),
                hd,
                len,
                buf,
            });
        }
        self.rope.as_ref().unwrap().buf.ptr
    }
}

/// Compile kernels.cu to a cubin for `arch` (e.g. sm_89).
fn compile(libs: &Libs, arch: &str) -> Result<Vec<u8>, String> {
    let rtc = &libs.rtc;
    let src = CString::new(SOURCE).unwrap();
    let mut prog = std::ptr::null_mut();
    let r = unsafe {
        (rtc.create_program)(
            &mut prog,
            src.as_ptr(),
            c"kernels.cu".as_ptr(),
            0,
            std::ptr::null(),
            std::ptr::null(),
        )
    };
    if r != 0 {
        return Err(format!("nvrtcCreateProgram: {}", libs.rtc_error(r)));
    }
    let opts: Vec<CString> = [format!("--gpu-architecture={arch}"), "-std=c++17".into()]
        .into_iter()
        .map(|s| CString::new(s).unwrap())
        .collect();
    let optp: Vec<*const c_char> = opts.iter().map(|s| s.as_ptr()).collect();
    let r = unsafe { (rtc.compile_program)(prog, optp.len() as c_int, optp.as_ptr()) };
    let mut log_n = 0;
    unsafe { (rtc.get_program_log_size)(prog, &mut log_n) };
    let mut log = vec![0 as c_char; log_n.max(1)];
    unsafe { (rtc.get_program_log)(prog, log.as_mut_ptr()) };
    let log = bytes_to_string(&log);
    if r != 0 {
        unsafe { (rtc.destroy_program)(&mut prog) };
        return Err(format!(
            "CUDA kernel compile failed for {arch} ({}):\n{log}",
            libs.rtc_error(r)
        ));
    }
    let mut n = 0;
    unsafe { (rtc.get_cubin_size)(prog, &mut n) };
    let mut cubin = vec![0u8; n];
    let r = unsafe { (rtc.get_cubin)(prog, cubin.as_mut_ptr() as *mut c_char) };
    unsafe { (rtc.destroy_program)(&mut prog) };
    if r != 0 || n == 0 {
        return Err(format!("nvrtcGetCUBIN for {arch}: {}", libs.rtc_error(r)));
    }
    Ok(cubin)
}

impl Drop for Cuda {
    fn drop(&mut self) {
        self.pass = None;
        self.rope = None;
        unsafe { (self.cu().stream_synchronize)(self.inner.stream) };
        for p in self.plans.borrow().values() {
            self.destroy_plan(p);
        }
        unsafe {
            (self.inner.libs.lt.pref_destroy)(self.pref);
            (self.inner.libs.lt.destroy)(self.lt);
            for p in [&self.staging, &self.readback] {
                if !p.ptr.is_null() {
                    (self.cu().mem_free_host)(p.ptr as *mut c_void);
                }
            }
            (self.cu().module_unload)(self.module);
        }
    }
}

impl Backend for Cuda {
    type W = CudaW;
    type T = CudaT;

    fn name(&self) -> &'static str {
        "cuda"
    }

    fn upload(&mut self, data: &[f32], rows: usize, cols: usize, kind: WeightKind) -> CudaW {
        assert_eq!(data.len(), rows * cols);
        let host: Vec<u8> = match kind {
            WeightKind::Matrix | WeightKind::Table => data
                .iter()
                .flat_map(|&v| f32_to_f16(v).to_le_bytes())
                .collect(),
            WeightKind::Vector => data.iter().flat_map(|v| v.to_le_bytes()).collect(),
        };
        let ptr = self
            .inner
            .alloc(host.len())
            .unwrap_or_else(|e| panic!("{e}"));
        unsafe {
            self.inner.check(
                (self.cu().memcpy_htod)(ptr, host.as_ptr() as *const c_void, host.len()),
                "upload",
            )
        };
        CudaW {
            buf: Buf {
                ptr,
                bytes: host.len(),
                pooled: false,
                inner: self.inner.clone(),
            },
            rows,
            cols,
        }
    }

    fn begin(&mut self, batch: &Batch) {
        self.pass = None;
        self.pass_started = Some(std::time::Instant::now());
        self.tokens = batch.tokens();
        let mut tiles: Vec<u32> = vec![];
        let mut longest = 0;
        for w in batch.starts.windows(2) {
            let (start, len) = (w[0], w[1] - w[0]);
            longest = longest.max(len as usize);
            for q0 in (0..len).step_by(if self.mma { FQ } else { QT }) {
                tiles.extend([start, len, q0]);
            }
        }
        // one staging copy of ids | positions | tiles (the previous pass has finished with it)
        let parts: [&[u32]; 3] = [&batch.ids, &batch.positions, &tiles];
        let total: usize = parts.iter().map(|p| p.len() * 4).sum();
        let mut staging = std::mem::replace(&mut self.staging, Pinned::empty());
        self.grow(&mut staging, total);
        let mut off = 0;
        let mut bufs = vec![];
        for p in parts {
            let n = p.len() * 4;
            let b = self.scratch(n);
            unsafe {
                std::ptr::copy_nonoverlapping(p.as_ptr() as *const u8, staging.ptr.add(off), n);
                self.inner.check(
                    (self.cu().memcpy_htod_async)(
                        b.ptr,
                        staging.ptr.add(off) as *const c_void,
                        n,
                        self.inner.stream,
                    ),
                    "batch upload",
                );
            }
            off += n;
            bufs.push(b);
        }
        self.staging = staging;
        let tiles_b = bufs.pop().unwrap();
        let pos = bufs.pop().unwrap();
        let ids = bufs.pop().unwrap();
        self.pass = Some(Pass {
            ids,
            pos,
            tiles: tiles_b,
            n_tiles: tiles.len() / 3,
            longest,
        });
    }

    fn embed(&mut self, word: &CudaW, pos: Option<&CudaW>, type0: Option<&CudaW>) -> CudaT {
        let c = word.cols;
        let out = self.tensor(self.tokens, c, true, false);
        let pass = self.pass.as_ref().expect("begin() not called");
        let (cols, has_pos, has_type) =
            (c as c_int, pos.is_some() as c_int, type0.is_some() as c_int);
        let ptab = pos.map(|w| w.buf.ptr).unwrap_or(self.dummy.ptr);
        let ttab = type0.map(|w| w.buf.ptr).unwrap_or(self.dummy.ptr);
        self.launch(
            "embed",
            (c.div_ceil(256), self.tokens),
            256,
            0,
            &mut args![
                pass.ids.ptr,
                pass.pos.ptr,
                word.buf.ptr,
                ptab,
                ttab,
                out.f32(),
                cols,
                has_pos,
                has_type
            ],
        );
        out
    }

    fn linear(
        &mut self,
        x: &CudaT,
        w: &CudaW,
        b: Option<&CudaW>,
        act: Act,
        residual: Option<&CudaT>,
    ) -> CudaT {
        assert_eq!(x.cols, w.cols, "linear: in features");
        let (m, n) = (x.rows, w.rows);
        // matmul input: the f16 copy a previous kernel wrote, else convert now
        let cast;
        let x16 = match &x.f16 {
            Some(h) => h.ptr,
            None => {
                cast = self.scratch(m * x.cols * 2);
                self.convert("to_f16", x.f32(), cast.ptr, m * x.cols);
                cast.ptr
            }
        };
        let gelu = act == Act::Gelu;
        // a GELU output feeds the next matmul: give it an f16 copy
        let out = self.tensor(m, n, true, gelu);
        let bias = b.map(|b| b.buf.ptr);
        if let (true, Some(r)) = (gelu, residual) {
            // not used by any model: matmul + bias, then GELU and the residual in one pass
            self.gemm(x16, w, out.f32(), m, bias, None);
            let (res, y16) = (r.f32(), out.f16.as_ref().unwrap().ptr);
            let (nn, z, one) = (n as c_int, 0 as c_int, 1 as c_int);
            self.launch(
                "epilogue",
                (n.div_ceil(256), m),
                256,
                0,
                &mut args![out.f32(), self.dummy.ptr, res, y16, nn, z, one, one],
            );
            return out;
        }
        self.gemm(x16, w, out.f32(), m, bias, residual.map(|r| r.f32()));
        if gelu {
            let y16 = out.f16.as_ref().unwrap().ptr;
            let (nn, z, one) = (n as c_int, 0 as c_int, 1 as c_int);
            self.launch(
                "epilogue",
                (n.div_ceil(256), m),
                256,
                0,
                &mut args![
                    out.f32(),
                    self.dummy.ptr,
                    self.dummy.ptr,
                    y16,
                    nn,
                    z,
                    z,
                    one
                ],
            );
        }
        out
    }

    fn layer_norm(&mut self, x: &CudaT, g: &CudaW, b: &CudaW, eps: f32) -> CudaT {
        let out = self.tensor(x.rows, x.cols, true, true);
        let cols = x.cols as c_int;
        let o16 = out.f16.as_ref().unwrap().ptr;
        self.launch(
            "layer_norm",
            (x.rows, 1),
            256,
            0,
            &mut args![x.f32(), g.buf.ptr, b.buf.ptr, out.f32(), o16, cols, eps],
        );
        out
    }

    fn rms_norm(&mut self, x: &CudaT, g: &CudaW, eps: f32) -> CudaT {
        let out = self.tensor(x.rows, x.cols, true, true);
        let cols = x.cols as c_int;
        let o16 = out.f16.as_ref().unwrap().ptr;
        self.launch(
            "rms_norm",
            (x.rows, 1),
            256,
            0,
            &mut args![x.f32(), g.buf.ptr, out.f32(), o16, cols, eps],
        );
        out
    }

    fn qk_norm_rope(
        &mut self,
        qkv: &CudaT,
        s: &AttnShape,
        qn: &CudaW,
        kn: &CudaW,
        eps: f32,
        theta: f32,
    ) {
        assert!(
            s.head_dim.is_multiple_of(64) && s.head_dim <= 256,
            "qk_norm_rope: head_dim must be a multiple of 64, ≤ 256"
        );
        let longest = self.pass.as_ref().expect("begin() not called").longest;
        let table = self.rope_table(s.head_dim, theta, longest);
        let pass = self.pass.as_ref().unwrap();
        let (stride, n_q, nh, hd) = (
            qkv.cols as c_int,
            s.n_q as c_int,
            (s.n_q + s.n_kv) as c_int,
            s.head_dim as c_int,
        );
        self.launch(
            "qk_norm_rope",
            ((s.n_q + s.n_kv).div_ceil(4), qkv.rows),
            128,
            0,
            &mut args![
                qkv.f32(),
                pass.pos.ptr,
                qn.buf.ptr,
                kn.buf.ptr,
                table,
                stride,
                n_q,
                nh,
                hd,
                eps
            ],
        );
    }

    fn attention(&mut self, qkv: &CudaT, s: &AttnShape) -> CudaT {
        assert_eq!(qkv.cols, s.qkv_cols());
        let hd = s.head_dim;
        if !matches!(hd, 32 | 64 | 128) {
            panic!("attention: head_dim {hd} unsupported on CUDA (32, 64, 128)");
        }
        let (kernel, smem) = match (self.mma, hd) {
            (true, 32) => ("fattn_32", fattn_smem(32)),
            (true, 64) => ("fattn_64", fattn_smem(64)),
            (true, _) => ("fattn_128", fattn_smem(128)),
            (false, 32) => ("attention_32", attn_smem(32)),
            (false, 64) => ("attention_64", attn_smem(64)),
            (false, _) => ("attention_128", attn_smem(128)),
        };
        // only the output projection reads it
        let out = self.tensor(qkv.rows, s.n_q * hd, false, true);
        let pass = self.pass.as_ref().expect("begin() not called");
        let (n_q, n_kv, causal, scale) = (
            s.n_q as c_int,
            s.n_kv as c_int,
            s.causal as c_int,
            1.0 / (hd as f32).sqrt(),
        );
        let o16 = out.f16.as_ref().unwrap().ptr;
        self.launch(
            kernel,
            (pass.n_tiles, s.n_q),
            128,
            smem,
            &mut args![qkv.f32(), pass.tiles.ptr, o16, n_q, n_kv, causal, scale],
        );
        out
    }

    fn silu_mul(&mut self, x: &CudaT) -> CudaT {
        let i = x.cols / 2;
        // only the down projection reads it
        let out = self.tensor(x.rows, i, false, true);
        let ii = i as c_int;
        let o16 = out.f16.as_ref().unwrap().ptr;
        self.launch(
            "silu_mul",
            (i.div_ceil(256), x.rows),
            256,
            0,
            &mut args![x.f32(), o16, ii],
        );
        out
    }

    fn finish<R>(&mut self, x: &CudaT, f: impl FnOnce(&[f32], usize) -> R) -> R {
        let elems = x.rows * x.cols;
        let n = elems * 4;
        let conv;
        let src = match &x.f32 {
            Some(b) => b.ptr,
            None => {
                conv = self.scratch(n);
                self.convert("to_f32", x.f16.as_ref().unwrap().ptr, conv.ptr, elems);
                conv.ptr
            }
        };
        let enqueued = self.pass_started.map(|t| t.elapsed().as_secs_f64() * 1e3);
        let mut rb = std::mem::replace(&mut self.readback, Pinned::empty());
        self.grow(&mut rb, n);
        unsafe {
            self.inner.check(
                (self.cu().memcpy_dtoh_async)(rb.ptr as *mut c_void, src, n, self.inner.stream),
                "readback",
            );
            self.inner.check(
                (self.cu().stream_synchronize)(self.inner.stream),
                "forward pass",
            );
        }
        if self.timing {
            eprintln!(
                "cuda pass: {} tokens, enqueue {:.2} ms, wall {:.2} ms",
                x.rows,
                enqueued.unwrap_or(0.0),
                self.pass_started
                    .map(|t| t.elapsed().as_secs_f64() * 1e3)
                    .unwrap_or(0.0)
            );
        }
        let data = unsafe { std::slice::from_raw_parts(rb.ptr as *const f32, elems) };
        let r = f(data, x.cols);
        self.readback = rb;
        self.pass = None;
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

    fn gpu() -> Option<Cuda> {
        match Cuda::new() {
            Ok(g) => Some(g),
            Err(e) if e.contains("could not load") || e.contains("no CUDA device") => {
                eprintln!("skip: {e}");
                None
            }
            Err(e) => panic!("{e}"),
        }
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

    fn block_case<B: Backend>(
        b: &mut B,
        causal: bool,
        bt: &Batch,
        hd: usize,
        n_q: usize,
        n_kv: usize,
    ) -> Vec<f32> {
        let hid = 128;
        let s = AttnShape {
            n_q,
            n_kv,
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
            &h(rnd(2 * hid * n_q * hd, 17, 0.1)),
            2 * hid,
            n_q * hd,
            WeightKind::Matrix,
        );
        b.begin(bt);
        let e = b.embed(&wt, Some(&wt), Some(&wt));
        let x = b.layer_norm(&e, &g, &bb, 1e-12);
        let x = b.rms_norm(&x, &g, 1e-6);
        let qkv = b.linear(&x, &wqkv, None, Act::None, None);
        if hd.is_multiple_of(64) {
            b.qk_norm_rope(&qkv, &s, &qn, &kn, 1e-6, 1e6);
        }
        let a = b.attention(&qkv, &s);
        let o = b.linear(&a, &wo, None, Act::None, None);
        let m = b.silu_mul(&o);
        b.finish(&m, |d, _| d.to_vec())
    }

    #[test]
    fn kernels_match_cpu() {
        let Some(mut gpu) = gpu() else { return };
        let mut cpu = Cpu::new();
        // f16 matmul inputs: the tolerance covers rounding the activations to f16
        for (n, k, act, bias, res) in [
            (100, 64, Act::None, false, false),
            (64, 96, Act::Gelu, true, false),
            (129, 32, Act::None, true, true),
            (100, 64, Act::None, false, true),
            (384, 384, Act::Gelu, true, true),
        ] {
            let c = gemm_case(&mut cpu, n, k, act, bias, res);
            let g = gemm_case(&mut gpu, n, k, act, bias, res);
            close(
                &g,
                &c,
                3e-3,
                &format!("gemm n={n} k={k} {act:?} bias={bias} res={res}"),
            );
        }
        let a: Vec<u32> = (0..1100).map(|i| i % 200).collect();
        let b2: Vec<u32> = (0..300).map(|i| (i * 7) % 200).collect();
        let long = Batch::new(&[&a, &b2]);
        for mma in [true, false] {
            gpu.mma = mma;
            for (hd, n_q, n_kv) in [(64, 4, 2), (128, 4, 1), (32, 4, 4)] {
                for causal in [false, true] {
                    for (name, bt) in [("short", batch()), ("long", long.clone())] {
                        let c = block_case(&mut cpu, causal, &bt, hd, n_q, n_kv);
                        let g = block_case(&mut gpu, causal, &bt, hd, n_q, n_kv);
                        // silu_mul's output is f16 on CUDA
                        close(
                            &g,
                            &c,
                            1e-2,
                            &format!(
                                "block mma={mma} hd={hd} q={n_q} kv={n_kv} causal={causal} {name}"
                            ),
                        );
                    }
                }
            }
        }
    }

    /// cargo test --release --lib cuda_gemm_speed -- --ignored --nocapture
    #[test]
    #[ignore]
    fn cuda_gemm_speed() {
        let Some(mut gpu) = gpu() else { return };
        for (m, n, k) in [
            (27, 4096, 1024),
            (4096, 4096, 1024),
            (4096, 6144, 1024),
            (4096, 1024, 3072),
        ] {
            let ids: Vec<u32> = (0..m as u32).map(|i| i % 200).collect();
            let bt = Batch::new(&[&ids]);
            let table = gpu.upload(&rnd(200 * k, 1, 1.0), 200, k, WeightKind::Table);
            let w = gpu.upload(&rnd(n * k, 3, 0.2), n, k, WeightKind::Matrix);
            let reps = if m < 100 { 200 } else { 50 };
            gpu.begin(&bt);
            let x = gpu.embed(&table, None, None);
            let y = gpu.linear(&x, &w, None, Act::None, None);
            gpu.finish(&y, |_, _| ());
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
                "gemm {m}x{n}x{k}: {:.3} ms  {:.1} TFLOPS",
                s * 1e3,
                2.0 * (m * n * k) as f64 / s / 1e12
            );
        }
    }
}
