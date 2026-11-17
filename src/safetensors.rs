//! Read-only safetensors over mmap (libc FFI; std has no mmap).

use crate::json::Json;
use std::collections::HashMap;
use std::fs::File;
use std::os::fd::AsRawFd;
use std::path::Path;

unsafe extern "C" {
    fn mmap(addr: *mut u8, len: usize, prot: i32, flags: i32, fd: i32, off: i64) -> *mut u8;
    fn munmap(addr: *mut u8, len: usize) -> i32;
}
const PROT_READ: i32 = 1;
const MAP_PRIVATE: i32 = 2;

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Dtype {
    F32,
    F16,
    BF16,
    I64,
    Other,
}

#[derive(Clone, Debug)]
pub struct TensorInfo {
    pub dtype: Dtype,
    pub shape: Vec<usize>,
    pub start: usize,
    pub end: usize,
}

pub struct SafeTensors {
    ptr: *mut u8,
    len: usize,
    data_start: usize,
    pub tensors: HashMap<String, TensorInfo>,
}

unsafe impl Send for SafeTensors {}
unsafe impl Sync for SafeTensors {}

impl Drop for SafeTensors {
    fn drop(&mut self) {
        unsafe { munmap(self.ptr, self.len) };
    }
}

impl SafeTensors {
    pub fn open(path: &Path) -> Result<SafeTensors, String> {
        let f = File::open(path).map_err(|e| format!("{}: {e}", path.display()))?;
        let len = f.metadata().map_err(|e| e.to_string())?.len() as usize;
        if len < 8 {
            return Err(format!("{}: too small for safetensors", path.display()));
        }
        let ptr = unsafe {
            mmap(
                std::ptr::null_mut(),
                len,
                PROT_READ,
                MAP_PRIVATE,
                f.as_raw_fd(),
                0,
            )
        };
        if ptr as isize == -1 || ptr.is_null() {
            return Err(format!("{}: mmap failed", path.display()));
        }
        let bytes = unsafe { std::slice::from_raw_parts(ptr, len) };
        let n = u64::from_le_bytes(bytes[..8].try_into().unwrap()) as usize;
        if 8 + n > len {
            unsafe { munmap(ptr, len) };
            return Err("safetensors header length out of range".into());
        }
        let header = std::str::from_utf8(&bytes[8..8 + n])
            .map_err(|_| "safetensors header is not UTF-8".to_string())?;
        let h = Json::parse(header)?;
        let mut tensors = HashMap::new();
        for (name, v) in h.as_obj().ok_or("safetensors header is not an object")? {
            if name == "__metadata__" {
                continue;
            }
            let dtype = match v.get("dtype").and_then(Json::as_str) {
                Some("F32") => Dtype::F32,
                Some("F16") => Dtype::F16,
                Some("BF16") => Dtype::BF16,
                Some("I64") => Dtype::I64,
                _ => Dtype::Other,
            };
            let shape = v
                .get("shape")
                .and_then(Json::as_arr)
                .ok_or("missing shape")?
                .iter()
                .map(|x| x.as_u64().unwrap_or(0) as usize)
                .collect();
            let off = v
                .get("data_offsets")
                .and_then(Json::as_arr)
                .ok_or("missing data_offsets")?;
            let (start, end) = (
                off[0].as_u64().unwrap_or(0) as usize,
                off[1].as_u64().unwrap_or(0) as usize,
            );
            if 8 + n + end > len || start > end {
                return Err(format!("tensor {name} out of range"));
            }
            tensors.insert(
                name.clone(),
                TensorInfo {
                    dtype,
                    shape,
                    start,
                    end,
                },
            );
        }
        Ok(SafeTensors {
            ptr,
            len,
            data_start: 8 + n,
            tensors,
        })
    }

    pub fn info(&self, name: &str) -> Result<&TensorInfo, String> {
        self.tensors
            .get(name)
            .ok_or_else(|| format!("missing tensor {name}"))
    }

    pub fn bytes(&self, t: &TensorInfo) -> &[u8] {
        unsafe {
            std::slice::from_raw_parts(self.ptr.add(self.data_start + t.start), t.end - t.start)
        }
    }

    /// Tensor converted to f32 (row-major), with its shape.
    pub fn f32(&self, name: &str) -> Result<(Vec<f32>, Vec<usize>), String> {
        let t = self.info(name)?;
        let b = self.bytes(t);
        let v: Vec<f32> = match t.dtype {
            Dtype::F32 => b
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
                .collect(),
            Dtype::F16 => b
                .chunks_exact(2)
                .map(|c| f16_to_f32(u16::from_le_bytes([c[0], c[1]])))
                .collect(),
            Dtype::BF16 => b
                .chunks_exact(2)
                .map(|c| f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16))
                .collect(),
            d => return Err(format!("tensor {name}: unsupported dtype {d:?}")),
        };
        let n: usize = t.shape.iter().product();
        if v.len() != n {
            return Err(format!("tensor {name}: size mismatch"));
        }
        Ok((v, t.shape.clone()))
    }
}

pub fn f16_to_f32(h: u16) -> f32 {
    let s = ((h >> 15) as u32) << 31;
    let e = ((h >> 10) & 0x1f) as u32;
    let m = (h & 0x3ff) as u32;
    let bits = if e == 0 {
        if m == 0 {
            s
        } else {
            // subnormal
            let mut e = 127 - 15 + 1;
            let mut m = m;
            while m & 0x400 == 0 {
                m <<= 1;
                e -= 1;
            }
            s | (e << 23) | ((m & 0x3ff) << 13)
        }
    } else if e == 31 {
        s | 0x7f80_0000 | (m << 13)
    } else {
        s | ((e + 127 - 15) << 23) | (m << 13)
    };
    f32::from_bits(bits)
}

/// Round-to-nearest-even f32 → f16 bits.
pub fn f32_to_f16(x: f32) -> u16 {
    let b = x.to_bits();
    let s = ((b >> 16) & 0x8000) as u16;
    let e = ((b >> 23) & 0xff) as i32;
    let m = b & 0x7f_ffff;
    if e == 255 {
        return s | 0x7c00 | if m != 0 { 0x200 } else { 0 };
    }
    let e16 = e - 127 + 15;
    if e16 >= 31 {
        return s | 0x7c00; // overflow → inf
    }
    if e16 <= 0 {
        if e16 < -10 {
            return s;
        }
        let m = m | 0x80_0000;
        let shift = (14 - e16) as u32;
        let half = 1u32 << (shift - 1);
        let mut r = m >> shift;
        let rem = m & ((1 << shift) - 1);
        if rem > half || (rem == half && r & 1 == 1) {
            r += 1;
        }
        return s | r as u16;
    }
    let mut r = ((e16 as u32) << 10) | (m >> 13);
    let rem = m & 0x1fff;
    if rem > 0x1000 || (rem == 0x1000 && r & 1 == 1) {
        r += 1;
    }
    s | r as u16
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn half_roundtrip() {
        for &x in &[0.0f32, 1.0, -2.5, 65504.0, 6.1e-5, 1e-7, 0.333333, -1234.5] {
            let y = f16_to_f32(f32_to_f16(x));
            assert!((x - y).abs() <= x.abs() * 1e-3 + 1e-7, "{x} {y}");
        }
        assert_eq!(f32_to_f16(1.0), 0x3c00);
        assert!(f16_to_f32(f32_to_f16(1e6)).is_infinite());
    }
}
