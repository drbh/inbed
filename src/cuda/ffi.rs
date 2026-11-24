//! The CUDA driver API, NVRTC and cuBLAS, loaded at runtime with `dlopen`, so the binary has no
//! link-time CUDA dependency and still runs (on the CPU) where there is no NVIDIA driver.
//!
//! Libraries are searched in `INBED_CUDA_LIB_PATH` (colon-separated directories), the directories
//! baked in at build time from the same variable, the NixOS driver and system profile paths, and
//! finally by soname through the dynamic loader.

use std::ffi::{CStr, CString, c_char, c_int, c_void};

pub type CuResult = c_int;
pub type CuDevice = c_int;
pub type CuContext = *mut c_void;
pub type CuModule = *mut c_void;
pub type CuFunction = *mut c_void;
pub type CuStream = *mut c_void;
pub type CuDevPtr = u64;
pub type NvrtcProgram = *mut c_void;
pub type LtHandle = *mut c_void;
pub type LtDesc = *mut c_void;
pub type LtLayout = *mut c_void;
pub type LtPref = *mut c_void;

pub const CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR: c_int = 75;
pub const CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR: c_int = 76;

pub const CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES: c_int = 8;

pub const CUBLAS_OP_N: i32 = 0;
pub const CUBLAS_OP_T: i32 = 1;
pub const CUDA_R_32F: c_int = 0;
pub const CUDA_R_16F: c_int = 2;
pub const CUBLAS_COMPUTE_32F: c_int = 68;
pub const CUBLASLT_MATMUL_DESC_TRANSA: c_int = 3;
pub const CUBLASLT_MATMUL_DESC_TRANSB: c_int = 4;
pub const CUBLASLT_MATMUL_DESC_EPILOGUE: c_int = 7;
pub const CUBLASLT_MATMUL_DESC_BIAS_POINTER: c_int = 8;
pub const CUBLASLT_EPILOGUE_DEFAULT: u32 = 1;
pub const CUBLASLT_EPILOGUE_BIAS: u32 = 4;
pub const CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES: c_int = 1;

/// `cublasLtMatmulAlgo_t`: opaque, 8 × u64
#[repr(C)]
#[derive(Clone, Copy)]
pub struct LtAlgo {
    pub data: [u64; 8],
}

/// `cublasLtMatmulHeuristicResult_t`
#[repr(C)]
#[derive(Clone, Copy)]
pub struct LtHeuristic {
    pub algo: LtAlgo,
    pub workspace_size: usize,
    pub state: c_int,
    pub waves_count: f32,
    pub reserved: [c_int; 4],
}

unsafe extern "C" {
    fn dlopen(filename: *const c_char, flag: c_int) -> *mut c_void;
    fn dlsym(handle: *mut c_void, symbol: *const c_char) -> *mut c_void;
    fn dlerror() -> *const c_char;
}
const RTLD_NOW: c_int = 2;
const RTLD_GLOBAL: c_int = 0x100;

fn search_dirs() -> Vec<String> {
    let mut dirs = vec![];
    if let Ok(p) = std::env::var("INBED_CUDA_LIB_PATH") {
        dirs.extend(p.split(':').filter(|s| !s.is_empty()).map(String::from));
    }
    if let Some(p) = option_env!("INBED_CUDA_LIB_PATH") {
        dirs.extend(p.split(':').filter(|s| !s.is_empty()).map(String::from));
    }
    dirs.extend(["/run/opengl-driver/lib", "/run/current-system/sw/lib"].map(String::from));
    dirs
}

/// Open the first of `names` found in the search path, then by soname.
fn open(names: &[&str]) -> Result<*mut c_void, String> {
    let mut tried = vec![];
    let dirs = search_dirs();
    let candidates = dirs
        .iter()
        .flat_map(|d| names.iter().map(move |n| format!("{d}/{n}")))
        .filter(|p| std::path::Path::new(p).exists())
        .chain(names.iter().map(|n| n.to_string()));
    for c in candidates {
        let cs = CString::new(c.clone()).unwrap();
        let h = unsafe { dlopen(cs.as_ptr(), RTLD_NOW | RTLD_GLOBAL) };
        if !h.is_null() {
            return Ok(h);
        }
        let e = unsafe { dlerror() };
        if !e.is_null() {
            tried.push(unsafe { CStr::from_ptr(e) }.to_string_lossy().into_owned());
        }
    }
    Err(format!(
        "could not load {} (set INBED_CUDA_LIB_PATH): {}",
        names[0],
        tried.last().map(String::as_str).unwrap_or("not found")
    ))
}

fn sym(h: *mut c_void, name: &str) -> Result<*mut c_void, String> {
    let cs = CString::new(name).unwrap();
    let p = unsafe { dlsym(h, cs.as_ptr()) };
    if p.is_null() {
        Err(format!("missing symbol {name}"))
    } else {
        Ok(p)
    }
}

/// Declares a table of function pointers resolved by name from one library.
macro_rules! api {
    ($name:ident { $( $field:ident = $sym:literal : fn($($arg:ty),*) -> $ret:ty; )* }) => {
        #[allow(non_snake_case, clippy::type_complexity)]
        pub struct $name {
            $( pub $field: unsafe extern "C" fn($($arg),*) -> $ret, )*
        }
        impl $name {
            fn load(h: *mut c_void) -> Result<$name, String> {
                Ok($name {
                    $( $field: unsafe { std::mem::transmute::<*mut c_void, unsafe extern "C" fn($($arg),*) -> $ret>(sym(h, $sym)?) }, )*
                })
            }
        }
    };
}

api!(Driver {
    init = "cuInit": fn(u32) -> CuResult;
    device_get = "cuDeviceGet": fn(*mut CuDevice, c_int) -> CuResult;
    device_get_count = "cuDeviceGetCount": fn(*mut c_int) -> CuResult;
    device_get_name = "cuDeviceGetName": fn(*mut c_char, c_int, CuDevice) -> CuResult;
    device_get_attribute = "cuDeviceGetAttribute": fn(*mut c_int, c_int, CuDevice) -> CuResult;
    primary_ctx_retain = "cuDevicePrimaryCtxRetain": fn(*mut CuContext, CuDevice) -> CuResult;
    primary_ctx_release = "cuDevicePrimaryCtxRelease_v2": fn(CuDevice) -> CuResult;
    ctx_set_current = "cuCtxSetCurrent": fn(CuContext) -> CuResult;
    mem_alloc = "cuMemAlloc_v2": fn(*mut CuDevPtr, usize) -> CuResult;
    mem_free = "cuMemFree_v2": fn(CuDevPtr) -> CuResult;
    mem_alloc_host = "cuMemAllocHost_v2": fn(*mut *mut c_void, usize) -> CuResult;
    mem_free_host = "cuMemFreeHost": fn(*mut c_void) -> CuResult;
    mem_get_info = "cuMemGetInfo_v2": fn(*mut usize, *mut usize) -> CuResult;
    memcpy_htod_async = "cuMemcpyHtoDAsync_v2": fn(CuDevPtr, *const c_void, usize, CuStream) -> CuResult;
    memcpy_dtoh_async = "cuMemcpyDtoHAsync_v2": fn(*mut c_void, CuDevPtr, usize, CuStream) -> CuResult;
    memcpy_htod = "cuMemcpyHtoD_v2": fn(CuDevPtr, *const c_void, usize) -> CuResult;
    stream_create = "cuStreamCreate": fn(*mut CuStream, u32) -> CuResult;
    stream_destroy = "cuStreamDestroy_v2": fn(CuStream) -> CuResult;
    stream_synchronize = "cuStreamSynchronize": fn(CuStream) -> CuResult;
    module_load_data = "cuModuleLoadData": fn(*mut CuModule, *const c_void) -> CuResult;
    module_unload = "cuModuleUnload": fn(CuModule) -> CuResult;
    module_get_function = "cuModuleGetFunction": fn(*mut CuFunction, CuModule, *const c_char) -> CuResult;
    func_set_attribute = "cuFuncSetAttribute": fn(CuFunction, c_int, c_int) -> CuResult;
    launch_kernel = "cuLaunchKernel": fn(CuFunction, u32, u32, u32, u32, u32, u32, u32, CuStream, *mut *mut c_void, *mut *mut c_void) -> CuResult;
    get_error_string = "cuGetErrorString": fn(CuResult, *mut *const c_char) -> CuResult;
});

api!(Nvrtc {
    create_program = "nvrtcCreateProgram": fn(*mut NvrtcProgram, *const c_char, *const c_char, c_int, *const *const c_char, *const *const c_char) -> c_int;
    compile_program = "nvrtcCompileProgram": fn(NvrtcProgram, c_int, *const *const c_char) -> c_int;
    get_program_log_size = "nvrtcGetProgramLogSize": fn(NvrtcProgram, *mut usize) -> c_int;
    get_program_log = "nvrtcGetProgramLog": fn(NvrtcProgram, *mut c_char) -> c_int;
    get_cubin_size = "nvrtcGetCUBINSize": fn(NvrtcProgram, *mut usize) -> c_int;
    get_cubin = "nvrtcGetCUBIN": fn(NvrtcProgram, *mut c_char) -> c_int;
    destroy_program = "nvrtcDestroyProgram": fn(*mut NvrtcProgram) -> c_int;
    get_error_string = "nvrtcGetErrorString": fn(c_int) -> *const c_char;
});

api!(Lt {
    create = "cublasLtCreate": fn(*mut LtHandle) -> c_int;
    destroy = "cublasLtDestroy": fn(LtHandle) -> c_int;
    desc_create = "cublasLtMatmulDescCreate": fn(*mut LtDesc, c_int, c_int) -> c_int;
    desc_destroy = "cublasLtMatmulDescDestroy": fn(LtDesc) -> c_int;
    desc_set = "cublasLtMatmulDescSetAttribute": fn(LtDesc, c_int, *const c_void, usize) -> c_int;
    layout_create = "cublasLtMatrixLayoutCreate": fn(*mut LtLayout, c_int, u64, u64, i64) -> c_int;
    layout_destroy = "cublasLtMatrixLayoutDestroy": fn(LtLayout) -> c_int;
    pref_create = "cublasLtMatmulPreferenceCreate": fn(*mut LtPref) -> c_int;
    pref_destroy = "cublasLtMatmulPreferenceDestroy": fn(LtPref) -> c_int;
    pref_set = "cublasLtMatmulPreferenceSetAttribute": fn(LtPref, c_int, *const c_void, usize) -> c_int;
    heuristic = "cublasLtMatmulAlgoGetHeuristic": fn(LtHandle, LtDesc, LtLayout, LtLayout, LtLayout, LtLayout, LtPref, c_int, *mut LtHeuristic, *mut c_int) -> c_int;
    matmul = "cublasLtMatmul": fn(LtHandle, LtDesc, *const c_void, *const c_void, LtLayout, *const c_void, LtLayout, *const c_void, *const c_void, LtLayout, *mut c_void, LtLayout, *const LtAlgo, *mut c_void, usize, CuStream) -> c_int;
});

pub struct Libs {
    pub cu: Driver,
    pub rtc: Nvrtc,
    pub lt: Lt,
}

impl Libs {
    pub fn load() -> Result<Libs, String> {
        let cu = Driver::load(open(&["libcuda.so.1", "libcuda.so"])?)?;
        let rtc = Nvrtc::load(open(&["libnvrtc.so", "libnvrtc.so.13", "libnvrtc.so.12"])?)?;
        let lt = Lt::load(open(&[
            "libcublasLt.so",
            "libcublasLt.so.13",
            "libcublasLt.so.12",
        ])?)?;
        Ok(Libs { cu, rtc, lt })
    }

    pub fn cu_error(&self, r: CuResult) -> String {
        let mut s: *const c_char = std::ptr::null();
        unsafe { (self.cu.get_error_string)(r, &mut s) };
        if s.is_null() {
            format!("CUDA error {r}")
        } else {
            format!("{} ({r})", unsafe { CStr::from_ptr(s) }.to_string_lossy())
        }
    }

    pub fn rtc_error(&self, r: c_int) -> String {
        let s = unsafe { (self.rtc.get_error_string)(r) };
        if s.is_null() {
            format!("NVRTC error {r}")
        } else {
            unsafe { CStr::from_ptr(s) }.to_string_lossy().into_owned()
        }
    }
}
