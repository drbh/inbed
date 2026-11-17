//! The HTTP contract against an in-process server: MiniLM on the CPU backend.

use inbed::api::Api;
use inbed::engine::{Device, Engine, EngineConfig};
use inbed::json::Json;
use std::io::{BufRead, BufReader, Read, Write};
use std::net::TcpStream;
use std::sync::{Arc, OnceLock};

const MODEL: &str = "sentence-transformers/all-MiniLM-L6-v2";

fn server() -> Option<&'static str> {
    static ADDR: OnceLock<Option<String>> = OnceLock::new();
    ADDR.get_or_init(|| {
        let r = inbed::hub::resolve(MODEL).ok()?;
        let spec = inbed::model::Spec::load(&r.id, &r.dir, None).unwrap();
        let e = Engine::start(
            spec,
            EngineConfig {
                device: Device::Cpu,
                max_batch_tokens: 4096,
                queue: 8,
                ..Default::default()
            },
        )
        .unwrap();
        e.wait_ready().unwrap();
        let api = Arc::new(Api {
            engines: vec![e],
            max_inputs: 8,
        });
        let l = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = l.local_addr().unwrap().to_string();
        std::thread::spawn(move || inbed::http::serve(l, 64, Arc::new(move |r| api.handle(r))));
        Some(addr)
    })
    .as_deref()
}

trait Io: Read + Write {}
impl<T: Read + Write> Io for T {}

struct Conn {
    r: BufReader<Box<dyn Io>>,
}

impl Conn {
    fn new(addr: &str) -> Conn {
        Conn::from(Box::new(TcpStream::connect(addr).unwrap()))
    }
    fn from(s: Box<dyn Io>) -> Conn {
        Conn {
            r: BufReader::new(s),
        }
    }
    fn raw(&mut self, req: &str) -> (u16, String) {
        let (s, _, b) = self.raw_full(req);
        (s, String::from_utf8(b).unwrap())
    }
    fn raw_full(&mut self, req: &str) -> (u16, Vec<(String, String)>, Vec<u8>) {
        self.r.get_mut().write_all(req.as_bytes()).unwrap();
        let mut line = String::new();
        self.r.read_line(&mut line).unwrap();
        let status: u16 = line.split_whitespace().nth(1).unwrap().parse().unwrap();
        let mut len = 0;
        let mut headers = vec![];
        loop {
            let mut h = String::new();
            self.r.read_line(&mut h).unwrap();
            let h = h.trim_end();
            if h.is_empty() {
                break;
            }
            let (k, v) = h.split_once(':').unwrap();
            let (k, v) = (k.trim().to_ascii_lowercase(), v.trim().to_string());
            if k == "content-length" {
                len = v.parse().unwrap();
            }
            headers.push((k, v));
        }
        let mut body = vec![0; len];
        self.r.read_exact(&mut body).unwrap();
        (status, headers, body)
    }
    fn get(&mut self, path: &str) -> (u16, Json) {
        let (s, b) = self.raw(&format!("GET {path} HTTP/1.1\r\nHost: x\r\n\r\n"));
        (s, Json::parse(&b).unwrap_or(Json::Null))
    }
    /// POST, retrying 429s the way a client should (they mean "back off", not "failed").
    fn post_retry(&mut self, body: &str) -> Json {
        loop {
            let (s, j) = self.post(body);
            if s != 429 {
                assert_eq!(s, 200, "{j}");
                return j;
            }
            std::thread::sleep(std::time::Duration::from_millis(5));
        }
    }
    fn post(&mut self, body: &str) -> (u16, Json) {
        let (s, b) = self.raw(&format!("POST /v1/embeddings HTTP/1.1\r\nHost: x\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{body}", body.len()));
        (s, Json::parse(&b).unwrap())
    }
}

fn code(j: &Json) -> &str {
    j.get("error")
        .and_then(|e| e.get("code"))
        .and_then(Json::as_str)
        .unwrap_or("")
}

#[test]
fn contract() {
    let Some(addr) = server() else {
        eprintln!("skip: {MODEL} not cached");
        return;
    };
    // one keep-alive connection for everything below
    let mut c = Conn::new(addr);

    let (s, _) = c.get("/health");
    assert_eq!(s, 200);

    let (s, m) = c.get("/v1/models");
    assert_eq!(s, 200);
    let d = &m.get("data").unwrap().as_arr().unwrap()[0];
    assert_eq!(d.get("id").unwrap().as_str(), Some(MODEL));
    assert_eq!(d.get("dims").unwrap().as_u64(), Some(384));
    assert_eq!(d.get("allowed_dims").unwrap().as_arr().unwrap().len(), 1);
    assert_eq!(d.get("max_tokens").unwrap().as_u64(), Some(256));
    assert_eq!(d.get("dtype").unwrap().as_str(), Some("f32"));

    let (s, r) = c.post(&format!(
        r#"{{"model":"{MODEL}","input":["red fox","","blue bird"]}}"#
    ));
    assert_eq!(s, 200);
    let data = r.get("data").unwrap().as_arr().unwrap();
    assert_eq!(data.len(), 3);
    for (i, e) in data.iter().enumerate() {
        assert_eq!(e.get("index").unwrap().as_u64(), Some(i as u64));
        let v = e.get("embedding").unwrap().as_arr().unwrap();
        assert_eq!(v.len(), 384);
        let n: f64 = v.iter().map(|x| x.as_f64().unwrap().powi(2)).sum();
        assert!((n - 1.0).abs() < 1e-4);
    }
    // [CLS] red fox [SEP] + [CLS] [SEP] + [CLS] blue bird [SEP]
    assert_eq!(
        r.get("usage")
            .unwrap()
            .get("prompt_tokens")
            .unwrap()
            .as_u64(),
        Some(10)
    );

    // single string, base64, model omitted (only one loaded)
    let (s, r) = c.post(r#"{"input":"red fox","encoding_format":"base64"}"#);
    assert_eq!(s, 200);
    let b64 = r.get("data").unwrap().as_arr().unwrap()[0]
        .get("embedding")
        .unwrap()
        .as_str()
        .unwrap()
        .to_string();
    assert_eq!(b64.len(), 2048); // 384 f32 = 1536 bytes = 2048 base64 chars

    // long input is truncated, never an error
    let long = "word ".repeat(2000);
    let (s, r) = c.post(&format!(r#"{{"input":"{long}"}}"#));
    assert_eq!(s, 200);
    assert_eq!(
        r.get("usage")
            .unwrap()
            .get("prompt_tokens")
            .unwrap()
            .as_u64(),
        Some(256)
    );

    assert_eq!(
        code(&c.post(r#"{"model":"nope","input":"x"}"#).1),
        "model_not_found"
    );
    assert_eq!(c.post(r#"{"model":"nope","input":"x"}"#).0, 404);
    assert_eq!(c.post(r#"{"input":"x","dimensions":100}"#).0, 400);
    assert_eq!(c.post(r#"{"input":[]}"#).0, 400);
    assert_eq!(c.post(r#"{"input":"x","input_type":"bogus"}"#).0, 400);
    assert_eq!(c.post(r#"{"input":[[101,99999999]]}"#).0, 400);
    assert_eq!(c.post("not json").0, 400);
    let (s, r) = c.post(r#"{"input":["a","b","c","d","e","f","g","h","i"]}"#);
    assert_eq!((s, code(&r)), (413, "batch_too_large"));
    let (s, _) = c.post(r#"{"input":"x","input_type":"query","dimensions":384}"#);
    assert_eq!(s, 200);

    // chunked bodies are refused with 411 (and the connection closes)
    let mut c2 = Conn::new(addr);
    let (s, _) = c2.raw(
        "POST /v1/embeddings HTTP/1.1\r\nTransfer-Encoding: chunked\r\n\r\n5\r\nhello\r\n0\r\n\r\n",
    );
    assert_eq!(s, 411);
}

#[test]
fn concurrent_requests_match_single() {
    let Some(addr) = server() else { return };
    let texts = [
        "red fox",
        "blue bird",
        "The quick brown fox jumps over the lazy dog.",
        "Café crème",
        "中文测试",
    ];
    let single: Vec<String> = texts
        .iter()
        .map(|t| {
            Conn::new(addr)
                .post(&format!(r#"{{"input":"{t}"}}"#))
                .1
                .get("data")
                .unwrap()
                .to_string()
        })
        .collect();
    let hs: Vec<_> = (0..20)
        .map(|i| {
            let t = texts[i % texts.len()];
            std::thread::spawn(move || {
                (
                    i,
                    Conn::new(addr)
                        .post_retry(&format!(r#"{{"input":"{t}"}}"#))
                        .get("data")
                        .unwrap()
                        .to_string(),
                )
            })
        })
        .collect();
    for h in hs {
        let (i, got) = h.join().unwrap();
        // CPU backend is deterministic per sequence regardless of batch composition
        let a = Json::parse(&got).unwrap();
        let b = Json::parse(&single[i % texts.len()]).unwrap();
        let va = a.as_arr().unwrap()[0]
            .get("embedding")
            .unwrap()
            .as_arr()
            .unwrap()
            .iter()
            .map(|x| x.as_f64().unwrap())
            .collect::<Vec<_>>();
        let vb = b.as_arr().unwrap()[0]
            .get("embedding")
            .unwrap()
            .as_arr()
            .unwrap()
            .iter()
            .map(|x| x.as_f64().unwrap())
            .collect::<Vec<_>>();
        let dot: f64 = va.iter().zip(&vb).map(|(x, y)| x * y).sum();
        assert!(dot > 0.99999, "batched result differs: cos {dot}");
    }
}

fn header<'a>(h: &'a [(String, String)], k: &str) -> Option<&'a str> {
    h.iter().find(|(n, _)| n == k).map(|(_, v)| v.as_str())
}

fn post_req(body: &str, accept: &str) -> String {
    format!(
        "POST /v1/embeddings HTTP/1.1\r\nHost: x\r\nAccept: {accept}\r\nContent-Length: {}\r\n\r\n{body}",
        body.len()
    )
}

/// §3a.1 binary responses and §3a.6 timing headers; the vectors equal the JSON ones bit for bit.
#[test]
fn binary_responses() {
    let Some(addr) = server() else { return };
    let mut c = Conn::new(addr);
    let body = r#"{"input":["red fox","blue bird"],"input_type":"query"}"#;
    let (s, h, bin) = c.raw_full(&post_req(
        body,
        "application/octet-stream, application/json;q=0.5",
    ));
    assert_eq!(s, 200);
    assert_eq!(header(&h, "content-type"), Some("application/octet-stream"));
    assert_eq!(header(&h, "x-embedding-count"), Some("2"));
    assert_eq!(header(&h, "x-embedding-dims"), Some("384"));
    assert_eq!(header(&h, "x-embedding-dtype"), Some("f32"));
    assert_eq!(header(&h, "x-usage-prompt-tokens"), Some("8"));
    assert!(header(&h, "x-inbed-queue-us").is_some());
    assert!(
        header(&h, "x-inbed-compute-us")
            .unwrap()
            .parse::<u64>()
            .unwrap()
            > 0
    );
    assert_eq!(bin.len(), 2 * 384 * 4);
    let floats: Vec<f32> = bin
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
        .collect();

    let (s, h, js) = c.raw_full(&post_req(body, "application/json"));
    assert_eq!(s, 200);
    assert_eq!(header(&h, "content-type"), Some("application/json"));
    assert!(header(&h, "x-inbed-compute-us").is_some());
    let j = Json::parse(std::str::from_utf8(&js).unwrap()).unwrap();
    let from_json: Vec<f32> = j
        .get("data")
        .unwrap()
        .as_arr()
        .unwrap()
        .iter()
        .flat_map(|e| {
            e.get("embedding")
                .unwrap()
                .as_arr()
                .unwrap()
                .iter()
                .map(|x| x.as_f64().unwrap() as f32)
                .collect::<Vec<_>>()
        })
        .collect();
    assert_eq!(floats, from_json);

    // MRL truncation applies to binary too; errors stay JSON whatever Accept says
    let (s, h, bin) = c.raw_full(&post_req(
        r#"{"input":"x","dimensions":384}"#,
        "application/octet-stream",
    ));
    assert_eq!((s, bin.len()), (200, 384 * 4));
    assert_eq!(header(&h, "x-embedding-dims"), Some("384"));
    let (s, h, _) = c.raw_full(&post_req(r#"{"input":[]}"#, "application/octet-stream"));
    assert_eq!(s, 400);
    assert_eq!(header(&h, "content-type"), Some("application/json"));
}

/// §3a.2 Unix socket listener and §3a.4 connection backpressure, on a separate small server.
#[test]
fn unix_socket_and_connection_limit() {
    use std::os::unix::net::UnixStream;
    use std::sync::atomic::AtomicUsize;
    let Some(_) = server() else { return };
    let r = inbed::hub::resolve(MODEL).unwrap();
    let spec = inbed::model::Spec::load(&r.id, &r.dir, None).unwrap();
    let e = Engine::start(
        spec,
        EngineConfig {
            device: Device::Cpu,
            ..Default::default()
        },
    )
    .unwrap();
    e.wait_ready().unwrap();
    let api = Arc::new(Api {
        engines: vec![e],
        max_inputs: 8,
    });
    let dir = std::env::temp_dir().join(format!("inbed-test-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("inbed.sock");
    let path = path.to_str().unwrap().to_string();
    // a stale socket file is replaced
    drop(std::os::unix::net::UnixListener::bind(&path).unwrap());
    let l = inbed::http::bind_unix(&path, 0o600).unwrap();
    assert!(
        inbed::http::bind_unix(&path, 0o600).is_err(),
        "live socket must not be replaced"
    );
    inbed::http::spawn(
        l,
        inbed::http::Config {
            threads: 2,
            idle_timeout: None,
        },
        Arc::new(AtomicUsize::new(0)),
        Arc::new(move |r| api.handle(r)),
    );

    let mut a = Conn::from(Box::new(UnixStream::connect(&path).unwrap()));
    let (s, h, bin) = a.raw_full(&post_req(
        r#"{"input":"red fox"}"#,
        "application/octet-stream",
    ));
    assert_eq!((s, bin.len()), (200, 384 * 4));
    assert!(header(&h, "x-inbed-compute-us").is_some());
    let mut b = Conn::from(Box::new(UnixStream::connect(&path).unwrap()));
    assert_eq!(b.get("/health").0, 200);
    // both threads hold keep-alive connections: the third is refused at once, not queued
    let t = std::time::Instant::now();
    let mut c = Conn::from(Box::new(UnixStream::connect(&path).unwrap()));
    let (s, body) = c.raw("GET /health HTTP/1.1\r\nHost: x\r\n\r\n");
    assert_eq!(s, 503);
    assert!(body.contains("too_many_connections"));
    assert!(t.elapsed() < std::time::Duration::from_secs(2));
    // closing one frees its slot
    drop(b);
    std::thread::sleep(std::time::Duration::from_millis(50));
    let mut d = Conn::from(Box::new(UnixStream::connect(&path).unwrap()));
    assert_eq!(d.get("/health").0, 200);
    assert_eq!(a.get("/health").0, 200);
    let _ = std::fs::remove_dir_all(&dir);
}
