//! The few libc calls std doesn't expose: socket buffer sizes and thread scheduling class.

use std::os::fd::RawFd;

unsafe extern "C" {
    fn setsockopt(fd: i32, level: i32, name: i32, val: *const std::ffi::c_void, len: u32) -> i32;
    #[cfg(target_os = "macos")]
    fn pthread_set_qos_class_self_np(class: u32, relpri: i32) -> i32;
}

#[cfg(target_os = "macos")]
mod c {
    pub const SOL_SOCKET: i32 = 0xffff;
    pub const SO_SNDBUF: i32 = 0x1001;
    pub const SO_RCVBUF: i32 = 0x1002;
}
#[cfg(not(target_os = "macos"))]
mod c {
    pub const SOL_SOCKET: i32 = 1;
    pub const SO_SNDBUF: i32 = 7;
    pub const SO_RCVBUF: i32 = 8;
}

fn set_int(fd: RawFd, level: i32, name: i32, v: i32) -> bool {
    unsafe {
        setsockopt(
            fd,
            level,
            name,
            &v as *const i32 as *const _,
            std::mem::size_of::<i32>() as u32,
        ) == 0
    }
}

/// Raise send and receive buffers (macOS defaults to 8 KB on Unix sockets). Best effort.
pub fn socket_buffers(fd: RawFd, bytes: i32) {
    set_int(fd, c::SOL_SOCKET, c::SO_SNDBUF, bytes);
    set_int(fd, c::SOL_SOCKET, c::SO_RCVBUF, bytes);
}

/// Ask the scheduler to treat the current thread as latency-critical. On macOS this is the
/// user-interactive QoS class, which keeps it off efficiency cores and wakes it promptly.
/// Elsewhere a no-op. Best effort.
pub fn latency_critical_thread() {
    #[cfg(target_os = "macos")]
    unsafe {
        const QOS_CLASS_USER_INTERACTIVE: u32 = 0x21;
        pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0);
    }
}
