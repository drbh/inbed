//! Minimal HTTP/1.1 server over TCP or a Unix socket: a fixed pool of threads, keep-alive,
//! Content-Length bodies. Connections beyond the pool get an immediate 503.

use std::io::{BufRead, BufReader, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::os::fd::AsRawFd;
use std::os::unix::net::{UnixListener, UnixStream};
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;

/// A connected byte stream the server can read requests from and write responses to.
pub trait Stream: Read + Write + Send + Sized + 'static {
    fn try_clone(&self) -> std::io::Result<Self>;
    fn set_read_timeout(&self, d: Option<Duration>) -> std::io::Result<()>;
    /// Socket options for a freshly accepted connection.
    fn tune(&self);
}

const SOCKET_BUFFER: i32 = 4 << 20;

impl Stream for TcpStream {
    fn try_clone(&self) -> std::io::Result<Self> {
        TcpStream::try_clone(self)
    }
    fn set_read_timeout(&self, d: Option<Duration>) -> std::io::Result<()> {
        TcpStream::set_read_timeout(self, d)
    }
    fn tune(&self) {
        let _ = self.set_nodelay(true);
        crate::sys::socket_buffers(self.as_raw_fd(), SOCKET_BUFFER);
    }
}

impl Stream for UnixStream {
    fn try_clone(&self) -> std::io::Result<Self> {
        UnixStream::try_clone(self)
    }
    fn set_read_timeout(&self, d: Option<Duration>) -> std::io::Result<()> {
        UnixStream::set_read_timeout(self, d)
    }
    fn tune(&self) {
        crate::sys::socket_buffers(self.as_raw_fd(), SOCKET_BUFFER);
    }
}

pub trait Listener: Send + Sync + 'static {
    type S: Stream;
    fn accept_stream(&self) -> std::io::Result<Self::S>;
}

impl Listener for TcpListener {
    type S = TcpStream;
    fn accept_stream(&self) -> std::io::Result<TcpStream> {
        self.accept().map(|(s, _)| s)
    }
}

impl Listener for UnixListener {
    type S = UnixStream;
    fn accept_stream(&self) -> std::io::Result<UnixStream> {
        self.accept().map(|(s, _)| s)
    }
}

/// Bind a Unix socket at `path`, replacing a stale socket file but refusing a live one.
pub fn bind_unix(path: &str, mode: u32) -> Result<UnixListener, String> {
    use std::os::unix::fs::{FileTypeExt, PermissionsExt};
    if let Ok(m) = std::fs::symlink_metadata(path) {
        if !m.file_type().is_socket() {
            return Err(format!("{path} exists and is not a socket"));
        }
        if UnixStream::connect(path).is_ok() {
            return Err(format!("{path}: another server is listening"));
        }
        std::fs::remove_file(path).map_err(|e| format!("remove stale {path}: {e}"))?;
    }
    let l = UnixListener::bind(path).map_err(|e| format!("bind {path}: {e}"))?;
    std::fs::set_permissions(path, std::fs::Permissions::from_mode(mode))
        .map_err(|e| format!("chmod {path}: {e}"))?;
    Ok(l)
}

pub struct Request {
    pub method: String,
    pub path: String,
    pub headers: Vec<(String, String)>,
    pub body: Vec<u8>,
}

impl Request {
    pub fn header(&self, name: &str) -> Option<&str> {
        self.headers
            .iter()
            .find(|(k, _)| k.eq_ignore_ascii_case(name))
            .map(|(_, v)| v.as_str())
    }
}

pub struct Response {
    pub status: u16,
    pub content_type: &'static str,
    /// extra headers, written as given
    pub headers: Vec<(&'static str, String)>,
    pub body: Vec<u8>,
}

impl Response {
    pub fn json(status: u16, body: String) -> Response {
        Response {
            status,
            content_type: "application/json",
            headers: Vec::new(),
            body: body.into_bytes(),
        }
    }
    pub fn header(mut self, k: &'static str, v: impl ToString) -> Response {
        self.headers.push((k, v.to_string()));
        self
    }
    pub fn text(status: u16, body: &str) -> Response {
        Response {
            status,
            content_type: "text/plain; charset=utf-8",
            headers: Vec::new(),
            body: body.as_bytes().to_vec(),
        }
    }
}

fn reason(status: u16) -> &'static str {
    match status {
        200 => "OK",
        400 => "Bad Request",
        404 => "Not Found",
        405 => "Method Not Allowed",
        411 => "Length Required",
        413 => "Payload Too Large",
        429 => "Too Many Requests",
        431 => "Request Header Fields Too Large",
        500 => "Internal Server Error",
        503 => "Service Unavailable",
        _ => "",
    }
}

pub const MAX_BODY: usize = 64 << 20;
const MAX_HEADER: usize = 32 << 10;

enum ReadErr {
    Closed,
    Bad(u16, &'static str),
}

fn read_request<S: Stream>(r: &mut BufReader<S>) -> Result<Request, ReadErr> {
    let mut line = String::new();
    let mut total = 0;
    match r.read_line(&mut line) {
        Ok(0) | Err(_) => return Err(ReadErr::Closed),
        Ok(n) => total += n,
    }
    let mut parts = line.split_whitespace();
    let (Some(method), Some(target)) = (parts.next(), parts.next()) else {
        return Err(ReadErr::Bad(400, "bad request line"));
    };
    let path = target.split('?').next().unwrap_or("/").to_string();
    let method = method.to_string();
    let mut headers = Vec::new();
    loop {
        let mut h = String::new();
        match r.read_line(&mut h) {
            Ok(0) | Err(_) => return Err(ReadErr::Closed),
            Ok(n) => total += n,
        }
        if total > MAX_HEADER {
            return Err(ReadErr::Bad(431, "headers too large"));
        }
        let h = h.trim_end();
        if h.is_empty() {
            break;
        }
        if let Some((k, v)) = h.split_once(':') {
            headers.push((k.trim().to_string(), v.trim().to_string()));
        }
    }
    let mut req = Request {
        method,
        path,
        headers,
        body: Vec::new(),
    };
    if req
        .header("transfer-encoding")
        .is_some_and(|v| !v.eq_ignore_ascii_case("identity"))
    {
        return Err(ReadErr::Bad(
            411,
            "chunked request bodies are not supported; send Content-Length",
        ));
    }
    let len: usize = match req.header("content-length") {
        Some(v) => v
            .parse()
            .map_err(|_| ReadErr::Bad(400, "bad Content-Length"))?,
        None => 0,
    };
    if len > MAX_BODY {
        return Err(ReadErr::Bad(413, "request body too large"));
    }
    if req
        .header("expect")
        .is_some_and(|v| v.eq_ignore_ascii_case("100-continue"))
    {
        let _ = r.get_mut().write_all(b"HTTP/1.1 100 Continue\r\n\r\n");
    }
    req.body = vec![0; len];
    r.read_exact(&mut req.body).map_err(|_| ReadErr::Closed)?;
    Ok(req)
}

fn write_response<S: Write>(s: &mut S, resp: &Response, keep_alive: bool) -> std::io::Result<()> {
    use std::fmt::Write as _;
    let mut head = format!(
        "HTTP/1.1 {} {}\r\nContent-Type: {}\r\nContent-Length: {}\r\nConnection: {}\r\nServer: inbed\r\n",
        resp.status,
        reason(resp.status),
        resp.content_type,
        resp.body.len(),
        if keep_alive { "keep-alive" } else { "close" }
    );
    for (k, v) in &resp.headers {
        let _ = write!(head, "{k}: {v}\r\n");
    }
    head.push_str("\r\n");
    // one write for small responses
    if resp.body.len() < 64 << 10 {
        let mut buf = head.into_bytes();
        buf.extend_from_slice(&resp.body);
        s.write_all(&buf)
    } else {
        s.write_all(head.as_bytes())?;
        s.write_all(&resp.body)
    }
}

/// Server-wide settings.
#[derive(Clone, Copy, Debug)]
pub struct Config {
    /// connections served at once (one thread each); more get 503
    pub threads: usize,
    /// close a connection after this long without a request; `None` never
    pub idle_timeout: Option<Duration>,
}

impl Default for Config {
    fn default() -> Self {
        Config {
            threads: 32,
            idle_timeout: Some(Duration::from_secs(600)),
        }
    }
}

pub type Handler = Arc<dyn Fn(&Request) -> Response + Send + Sync>;

fn serve_conn<S: Stream>(
    stream: S,
    idle: Option<Duration>,
    handler: &(dyn Fn(&Request) -> Response + Sync),
) {
    stream.tune();
    let _ = stream.set_read_timeout(idle);
    let Ok(mut w) = stream.try_clone() else {
        return;
    };
    let mut r = BufReader::with_capacity(64 << 10, stream);
    loop {
        match read_request(&mut r) {
            Ok(req) => {
                let close = req
                    .header("connection")
                    .is_some_and(|v| v.eq_ignore_ascii_case("close"));
                let resp = handler(&req);
                if write_response(&mut w, &resp, !close).is_err() || close {
                    return;
                }
            }
            Err(ReadErr::Closed) => return,
            Err(ReadErr::Bad(code, msg)) => {
                let _ = write_response(
                    &mut w,
                    &crate::api::error(code, "invalid_request", msg),
                    false,
                );
                return;
            }
        }
    }
}

/// Over the connection limit: read the request (briefly) so the client sees the response rather
/// than a reset, answer 503 and close.
fn refuse<S: Stream>(stream: S, limit: usize) {
    let _ = stream.set_read_timeout(Some(Duration::from_millis(200)));
    let Ok(mut w) = stream.try_clone() else {
        return;
    };
    let mut r = BufReader::new(stream);
    let _ = read_request(&mut r);
    let _ = write_response(
        &mut w,
        &crate::api::error(
            503,
            "too_many_connections",
            &format!("inbed serves at most {limit} connections at once; use a smaller pool"),
        ),
        false,
    );
}

/// Serve `listener` on `cfg.threads + 1` threads. `active` counts connections being served and
/// may be shared between listeners so the limit is server-wide. The extra thread means someone is
/// always in `accept`, so connection `threads + 1` is refused at once instead of waiting in the
/// kernel backlog.
pub fn spawn<L: Listener>(listener: L, cfg: Config, active: Arc<AtomicUsize>, handler: Handler) {
    let listener = Arc::new(listener);
    let limit = cfg.threads.max(1);
    for i in 0..=limit {
        let l = listener.clone();
        let h = handler.clone();
        let active = active.clone();
        std::thread::Builder::new()
            .name(format!("http-{i}"))
            .spawn(move || {
                crate::sys::latency_critical_thread();
                loop {
                    let Ok(s) = l.accept_stream() else { continue };
                    if active.fetch_add(1, Ordering::AcqRel) >= limit {
                        active.fetch_sub(1, Ordering::AcqRel);
                        refuse(s, limit);
                    } else {
                        serve_conn(s, cfg.idle_timeout, &*h);
                        active.fetch_sub(1, Ordering::AcqRel);
                    }
                }
            })
            .expect("spawn http thread");
    }
}

/// TCP only, with default settings except the thread count; never returns.
pub fn serve(listener: TcpListener, threads: usize, handler: Handler) -> ! {
    spawn(
        listener,
        Config {
            threads,
            ..Config::default()
        },
        Arc::new(AtomicUsize::new(0)),
        handler,
    );
    loop {
        std::thread::park();
    }
}
