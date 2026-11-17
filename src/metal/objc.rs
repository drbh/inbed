//! Just enough Objective-C runtime FFI to drive Metal: no `objc`/`metal` crates.
//!
//! Every function here sends messages to raw object pointers; callers inside this crate only pass
//! objects Metal handed back, so the helpers are safe wrappers by convention.
//!
//! `objc_msgSend` is cast to a correctly typed `extern "C"` fn pointer per call site (arm64 has
//! no `_stret` variant; large structs like `MTLSize` go by reference per AAPCS64 automatically).

#![allow(clippy::not_unsafe_ptr_arg_deref, clippy::macro_metavars_in_unsafe)]

use std::ffi::{CStr, c_char, c_void};

pub type Id = *mut c_void;
pub type Sel = *const c_void;

#[link(name = "objc")]
unsafe extern "C" {
    fn objc_getClass(name: *const c_char) -> Id;
    fn sel_registerName(name: *const c_char) -> Sel;
    pub fn objc_msgSend();
    fn objc_autoreleasePoolPush() -> *mut c_void;
    fn objc_autoreleasePoolPop(p: *mut c_void);
}

#[link(name = "Metal", kind = "framework")]
unsafe extern "C" {
    pub fn MTLCreateSystemDefaultDevice() -> Id;
}

#[link(name = "Foundation", kind = "framework")]
unsafe extern "C" {}

pub fn class(name: &CStr) -> Id {
    unsafe { objc_getClass(name.as_ptr()) }
}

pub fn sel(name: &CStr) -> Sel {
    unsafe { sel_registerName(name.as_ptr()) }
}

/// `msg!(Ret; obj, c"selector:with:", a: TypeA, b: TypeB)`
#[macro_export]
#[allow(clippy::macro_metavars_in_unsafe)]
macro_rules! msg {
    ($ret:ty; $obj:expr, $sel:expr $(, $arg:expr => $t:ty)* $(,)?) => {{
        #[allow(unused_unsafe)]
        unsafe {
            let f: unsafe extern "C" fn($crate::metal::objc::Id, $crate::metal::objc::Sel $(, $t)*) -> $ret =
                std::mem::transmute($crate::metal::objc::objc_msgSend as *const ());
            f($obj, $crate::metal::objc::sel($sel) $(, $arg)*)
        }
    }};
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct MTLSize {
    pub width: usize,
    pub height: usize,
    pub depth: usize,
}

pub fn size(w: usize, h: usize, d: usize) -> MTLSize {
    MTLSize {
        width: w,
        height: h,
        depth: d,
    }
}

pub fn release(o: Id) {
    if !o.is_null() {
        msg!((); o, c"release");
    }
}

/// +1 NSString (caller releases).
pub fn nsstring(s: &str) -> Id {
    let alloc = msg!(Id; class(c"NSString"), c"alloc");
    msg!(Id; alloc, c"initWithBytes:length:encoding:", s.as_ptr() as *const c_void => *const c_void, s.len() => usize, 4usize => usize)
}

pub fn to_string(ns: Id) -> String {
    if ns.is_null() {
        return String::new();
    }
    let p = msg!(*const c_char; ns, c"UTF8String");
    if p.is_null() {
        String::new()
    } else {
        unsafe { CStr::from_ptr(p) }.to_string_lossy().into_owned()
    }
}

pub fn error_string(err: Id) -> String {
    if err.is_null() {
        return "unknown error".into();
    }
    to_string(msg!(Id; err, c"localizedDescription"))
}

pub struct AutoreleasePool(*mut c_void);

impl AutoreleasePool {
    pub fn new() -> AutoreleasePool {
        AutoreleasePool(unsafe { objc_autoreleasePoolPush() })
    }
}

impl Default for AutoreleasePool {
    fn default() -> Self {
        Self::new()
    }
}

impl Drop for AutoreleasePool {
    fn drop(&mut self) {
        unsafe { objc_autoreleasePoolPop(self.0) }
    }
}
