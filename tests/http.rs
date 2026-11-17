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
        std::thread::spawn(move || inbed::http::serve(l, 4, Arc::new(move |r| api.handle(r))));
        Some(addr)
    })
    .as_deref()
}

struct Conn {
    r: BufReader<TcpStream>,
}

impl Conn {
    fn new(addr: &str) -> Conn {
        Conn {
            r: BufReader::new(TcpStream::connect(addr).unwrap()),
        }
    }
    fn raw(&mut self, req: &str) -> (u16, String) {
        self.r.get_mut().write_all(req.as_bytes()).unwrap();
        let mut line = String::new();
        self.r.read_line(&mut line).unwrap();
        let status: u16 = line.split_whitespace().nth(1).unwrap().parse().unwrap();
        let mut len = 0;
        loop {
            let mut h = String::new();
            self.r.read_line(&mut h).unwrap();
            let h = h.trim_end();
            if h.is_empty() {
                break;
            }
            if let Some(v) = h.to_ascii_lowercase().strip_prefix("content-length:") {
                len = v.trim().parse().unwrap();
            }
        }
        let mut body = vec![0; len];
        self.r.read_exact(&mut body).unwrap();
        (status, String::from_utf8(body).unwrap())
    }
    fn get(&mut self, path: &str) -> (u16, Json) {
        let (s, b) = self.raw(&format!("GET {path} HTTP/1.1\r\nHost: x\r\n\r\n"));
        (s, Json::parse(&b).unwrap_or(Json::Null))
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
                        .post(&format!(r#"{{"input":"{t}"}}"#))
                        .1
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
