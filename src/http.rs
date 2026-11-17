//! Minimal HTTP/1.1 server on std::net: a fixed pool of threads, keep-alive, Content-Length bodies.

use std::io::{BufRead, BufReader, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::Arc;
use std::time::Duration;

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
    pub body: Vec<u8>,
}

impl Response {
    pub fn json(status: u16, body: String) -> Response {
        Response {
            status,
            content_type: "application/json",
            body: body.into_bytes(),
        }
    }
    pub fn text(status: u16, body: &str) -> Response {
        Response {
            status,
            content_type: "text/plain; charset=utf-8",
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

fn read_request(r: &mut BufReader<TcpStream>) -> Result<Request, ReadErr> {
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

fn write_response(s: &mut TcpStream, resp: &Response, keep_alive: bool) -> std::io::Result<()> {
    let head = format!(
        "HTTP/1.1 {} {}\r\nContent-Type: {}\r\nContent-Length: {}\r\nConnection: {}\r\nServer: inbed\r\n\r\n",
        resp.status,
        reason(resp.status),
        resp.content_type,
        resp.body.len(),
        if keep_alive { "keep-alive" } else { "close" }
    );
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

fn serve_conn(stream: TcpStream, handler: &(dyn Fn(&Request) -> Response + Sync)) {
    let _ = stream.set_nodelay(true);
    let _ = stream.set_read_timeout(Some(Duration::from_secs(30)));
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

/// Accept on `threads` threads; each handles one connection at a time.
pub fn serve(
    listener: TcpListener,
    threads: usize,
    handler: Arc<dyn Fn(&Request) -> Response + Send + Sync>,
) -> ! {
    let listener = Arc::new(listener);
    for i in 0..threads.max(1) {
        let l = listener.clone();
        let h = handler.clone();
        std::thread::Builder::new()
            .name(format!("http-{i}"))
            .spawn(move || {
                loop {
                    if let Ok((s, _)) = l.accept() {
                        serve_conn(s, &*h);
                    }
                }
            })
            .expect("spawn http thread");
    }
    loop {
        std::thread::park();
    }
}
