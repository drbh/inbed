//! OpenAI-compatible embeddings API.

use crate::engine::{Engine, Kind, SubmitError};
use crate::http::{Request, Response};
use crate::json::{self, Json};
use crate::model::truncate_dims;
use std::fmt::Write as _;
use std::sync::Arc;

pub struct Api {
    pub engines: Vec<Arc<Engine>>,
    pub max_inputs: usize,
}

pub fn error(status: u16, code: &str, message: &str) -> Response {
    let ty = match status {
        429 => "rate_limit_error",
        500..=599 => "server_error",
        _ => "invalid_request_error",
    };
    let mut s = String::from("{\"error\":{\"message\":");
    json::write_str(&mut s, message);
    s.push_str(",\"type\":");
    json::write_str(&mut s, ty);
    s.push_str(",\"code\":");
    json::write_str(&mut s, code);
    s.push_str("}}");
    Response::json(status, s)
}

const B64: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

pub fn base64(b: &[u8]) -> String {
    let mut s = String::with_capacity(b.len().div_ceil(3) * 4);
    for c in b.chunks(3) {
        let n = (c[0] as u32) << 16
            | (*c.get(1).unwrap_or(&0) as u32) << 8
            | *c.get(2).unwrap_or(&0) as u32;
        s.push(B64[(n >> 18) as usize & 63] as char);
        s.push(B64[(n >> 12) as usize & 63] as char);
        s.push(if c.len() > 1 {
            B64[(n >> 6) as usize & 63] as char
        } else {
            '='
        });
        s.push(if c.len() > 2 {
            B64[n as usize & 63] as char
        } else {
            '='
        });
    }
    s
}

/// True when the client prefers `application/octet-stream` over JSON, e.g.
/// `Accept: application/octet-stream, application/json;q=0.5`.
pub fn wants_binary(accept: Option<&str>) -> bool {
    let Some(a) = accept else { return false };
    let (mut bin, mut json) = (0.0f32, 0.0f32);
    for part in a.split(',') {
        let mut it = part.split(';');
        let ty = it.next().unwrap_or("").trim().to_ascii_lowercase();
        let q = it
            .filter_map(|p| p.trim().strip_prefix("q="))
            .find_map(|v| v.trim().parse::<f32>().ok())
            .unwrap_or(1.0);
        match ty.as_str() {
            "application/octet-stream" => bin = bin.max(q),
            "application/json" | "application/*" | "*/*" => json = json.max(q),
            _ => {}
        }
    }
    bin > 0.0 && bin >= json
}

impl Api {
    pub fn handle(&self, req: &Request) -> Response {
        match (req.method.as_str(), req.path.trim_end_matches('/')) {
            ("POST", "/v1/embeddings" | "/embeddings") => self.embeddings(req),
            ("GET", "/v1/models" | "/models") => self.models(),
            ("GET", "/health" | "/healthz") => self.health(),
            ("GET", "/metrics") => self.metrics(),
            ("GET", "" | "/") => Response::text(
                200,
                "inbed: POST /v1/embeddings, GET /v1/models, GET /health\n",
            ),
            (_, "/v1/embeddings" | "/v1/models" | "/health") => {
                error(405, "invalid_request", "method not allowed")
            }
            _ => error(
                404,
                "not_found",
                &format!("no route for {} {}", req.method, req.path),
            ),
        }
    }

    fn health(&self) -> Response {
        if let Some(e) = self
            .engines
            .iter()
            .find_map(|e| e.failure().map(|f| (e.spec.id.clone(), f)))
        {
            return error(
                503,
                "not_ready",
                &format!("model {} failed to load: {}", e.0, e.1),
            );
        }
        if self.engines.iter().all(|e| e.is_ready()) {
            Response::json(200, "{\"status\":\"ok\"}".into())
        } else {
            error(503, "not_ready", "models are still loading")
        }
    }

    fn models(&self) -> Response {
        let data: Vec<Json> = self
            .engines
            .iter()
            .map(|e| {
                let s = &e.spec;
                crate::obj! {
                    "id" => s.id.as_str(),
                    "object" => "model",
                    "owned_by" => "inbed",
                    "dims" => s.dims,
                    "allowed_dims" => Json::Arr(s.allowed_dims.iter().map(|&d| Json::from(d)).collect()),
                    "max_tokens" => s.max_tokens,
                    "dtype" => "f32",
                    "arch" => s.arch.name(),
                    "device" => format!("{:?}", e.device).to_lowercase(),
                    "ready" => e.is_ready(),
                }
            })
            .collect();
        Response::json(
            200,
            crate::obj! {"object" => "list", "data" => Json::Arr(data)}.to_string(),
        )
    }

    fn metrics(&self) -> Response {
        use std::sync::atomic::Ordering::Relaxed;
        let mut s = String::new();
        for e in &self.engines {
            let st = e.stats();
            let m = &e.spec.id;
            let _ = writeln!(
                s,
                "inbed_requests{{model=\"{m}\"}} {}",
                st.requests.load(Relaxed)
            );
            let _ = writeln!(
                s,
                "inbed_rejected{{model=\"{m}\"}} {}",
                st.rejected.load(Relaxed)
            );
            let _ = writeln!(
                s,
                "inbed_sequences{{model=\"{m}\"}} {}",
                st.sequences.load(Relaxed)
            );
            let _ = writeln!(
                s,
                "inbed_tokens{{model=\"{m}\"}} {}",
                st.tokens.load(Relaxed)
            );
            let _ = writeln!(
                s,
                "inbed_batches{{model=\"{m}\"}} {}",
                st.batches.load(Relaxed)
            );
            let _ = writeln!(
                s,
                "inbed_busy_seconds{{model=\"{m}\"}} {:.3}",
                st.busy_us.load(Relaxed) as f64 / 1e6
            );
            let _ = writeln!(
                s,
                "inbed_queued_tokens{{model=\"{m}\",kind=\"query\"}} {}\ninbed_queued_tokens{{model=\"{m}\",kind=\"document\"}} {}",
                e.queued_tokens(Kind::Query),
                e.queued_tokens(Kind::Document)
            );
            let _ = writeln!(
                s,
                "inbed_tokens_per_second{{model=\"{m}\"}} {:.0}",
                e.throughput()
            );
        }
        Response::text(200, &s)
    }

    fn engine(&self, name: Option<&str>) -> Result<&Arc<Engine>, Response> {
        match name {
            None if self.engines.len() == 1 => Ok(&self.engines[0]),
            None => Err(error(400, "invalid_request", "\"model\" is required")),
            Some(n) => self
                .engines
                .iter()
                .find(|e| e.spec.id == n || e.spec.id.eq_ignore_ascii_case(n))
                .ok_or_else(|| {
                    let ids: Vec<&str> = self.engines.iter().map(|e| e.spec.id.as_str()).collect();
                    error(
                        404,
                        "model_not_found",
                        &format!("model {n:?} is not loaded (available: {})", ids.join(", ")),
                    )
                }),
        }
    }

    fn embeddings(&self, req: &Request) -> Response {
        let Ok(body) = std::str::from_utf8(&req.body) else {
            return error(400, "invalid_request", "body is not UTF-8");
        };
        let j = match Json::parse(body) {
            Ok(j) => j,
            Err(e) => return error(400, "invalid_request", &e),
        };
        if j.as_obj().is_none() {
            return error(400, "invalid_request", "body must be a JSON object");
        }
        let model = match j.get("model") {
            None | Some(Json::Null) => None,
            Some(Json::Str(s)) => Some(s.as_str()),
            Some(_) => return error(400, "invalid_request", "\"model\" must be a string"),
        };
        let eng = match self.engine(model) {
            Ok(e) => e,
            Err(r) => return r,
        };
        let spec = &eng.spec;

        let query = match j.get("input_type").map(|v| v.as_str()) {
            None | Some(Some("document" | "passage" | "search_document")) => false,
            Some(Some("query" | "search_query")) => true,
            Some(None) if j.get("input_type").is_some_and(Json::is_null) => false,
            _ => {
                return error(
                    400,
                    "invalid_request",
                    "\"input_type\" must be \"query\" or \"document\"",
                );
            }
        };
        let dims = match j.get("dimensions") {
            None | Some(Json::Null) => spec.dims,
            Some(v) => match v.as_u64() {
                Some(d) if spec.allowed_dims.contains(&(d as usize)) => d as usize,
                _ => {
                    return error(
                        400,
                        "invalid_request",
                        &format!(
                            "\"dimensions\" must be one of {:?} for {}",
                            spec.allowed_dims, spec.id
                        ),
                    );
                }
            },
        };
        let b64 = match j.get("encoding_format").map(|v| v.as_str()) {
            None | Some(Some("float")) => false,
            Some(Some("base64")) => true,
            Some(None) if j.get("encoding_format").is_some_and(Json::is_null) => false,
            _ => {
                return error(
                    400,
                    "invalid_request",
                    "\"encoding_format\" must be \"float\" or \"base64\"",
                );
            }
        };

        // input: string | [string] | [int] | [[int]]
        let seqs: Vec<Vec<u32>> = match j.get("input") {
            Some(Json::Str(s)) => vec![eng.tokenize(s, query)],
            Some(Json::Arr(a)) if a.is_empty() => {
                return error(400, "invalid_request", "\"input\" must not be empty");
            }
            Some(Json::Arr(a)) if a.len() > self.max_inputs => {
                return error(
                    413,
                    "batch_too_large",
                    &format!(
                        "{} inputs exceeds the limit of {}",
                        a.len(),
                        self.max_inputs
                    ),
                );
            }
            Some(Json::Arr(a)) if a.iter().all(|x| matches!(x, Json::Str(_))) => a
                .iter()
                .map(|x| eng.tokenize(x.as_str().unwrap(), query))
                .collect(),
            Some(Json::Arr(a)) if a.iter().all(|x| matches!(x, Json::Num(_))) => {
                match token_ids(a, spec) {
                    Some(ids) => vec![ids],
                    None => {
                        return error(
                            400,
                            "invalid_request",
                            "token ids must be integers in [0, vocab size)",
                        );
                    }
                }
            }
            Some(Json::Arr(a)) if a.iter().all(|x| matches!(x, Json::Arr(_))) => {
                let mut out = vec![];
                for x in a {
                    match token_ids(x.as_arr().unwrap(), spec) {
                        Some(ids) if !ids.is_empty() => out.push(ids),
                        _ => {
                            return error(
                                400,
                                "invalid_request",
                                "token id arrays must be non-empty arrays of integers in [0, vocab size)",
                            );
                        }
                    }
                }
                out
            }
            None => return error(400, "invalid_request", "\"input\" is required"),
            _ => {
                return error(
                    400,
                    "invalid_request",
                    "\"input\" must be a string or an array of strings",
                );
            }
        };
        let tokens: usize = seqs.iter().map(Vec::len).sum();

        let kind = if query { Kind::Query } else { Kind::Document };
        let ticket = match eng.submit(seqs, kind) {
            Ok(t) => t,
            Err(SubmitError::Busy { retry_after_ms }) => {
                return error(
                    429,
                    "rate_limited",
                    "the embedding queue is over its wait limit; retry later",
                )
                .header("Retry-After", retry_after_ms.div_ceil(1000).max(1));
            }
            Err(SubmitError::NotReady(m)) => return error(503, "not_ready", &m),
            Err(SubmitError::Failed(m)) => return error(503, "not_ready", &m),
        };
        let out = match ticket.wait() {
            Ok(v) => v,
            Err(e) => return error(500, "internal_error", &e),
        };
        let n = out.vecs.len();
        let vecs = out.vecs.into_iter().map(|v| truncate_dims(v, dims));

        let resp = if wants_binary(req.header("accept")) {
            let mut body = Vec::with_capacity(n * dims * 4);
            for v in vecs {
                for x in v {
                    body.extend_from_slice(&x.to_le_bytes());
                }
            }
            Response {
                status: 200,
                content_type: "application/octet-stream",
                headers: Vec::new(),
                body,
            }
            .header("X-Embedding-Count", n)
            .header("X-Embedding-Dims", dims)
            .header("X-Embedding-Dtype", "f32")
            .header("X-Usage-Prompt-Tokens", tokens)
        } else {
            let mut s = String::with_capacity(64 + n * dims * if b64 { 6 } else { 12 });
            s.push_str("{\"object\":\"list\",\"model\":");
            json::write_str(&mut s, &spec.id);
            s.push_str(",\"data\":[");
            for (i, v) in vecs.enumerate() {
                if i > 0 {
                    s.push(',');
                }
                let _ = write!(s, "{{\"object\":\"embedding\",\"index\":{i},\"embedding\":");
                if b64 {
                    let bytes: Vec<u8> = v.iter().flat_map(|x| x.to_le_bytes()).collect();
                    s.push('"');
                    s.push_str(&base64(&bytes));
                    s.push('"');
                } else {
                    s.push('[');
                    for (k, x) in v.iter().enumerate() {
                        if k > 0 {
                            s.push(',');
                        }
                        json::write_f32(&mut s, *x);
                    }
                    s.push(']');
                }
                s.push('}');
            }
            let _ = write!(
                s,
                "],\"usage\":{{\"prompt_tokens\":{tokens},\"total_tokens\":{tokens}}}}}"
            );
            Response::json(200, s)
        };
        resp.header("X-Inbed-Queue-Us", out.queue_us)
            .header("X-Inbed-Compute-Us", out.compute_us)
    }
}

fn token_ids(a: &[Json], spec: &crate::model::Spec) -> Option<Vec<u32>> {
    let mut v: Vec<u32> = a
        .iter()
        .map(|x| {
            x.as_u64()
                .filter(|&n| (n as usize) < spec.vocab)
                .map(|n| n as u32)
        })
        .collect::<Option<_>>()?;
    v.truncate(spec.max_tokens);
    Some(v)
}

#[cfg(test)]
mod tests {
    #[test]
    fn accept_negotiation() {
        use super::wants_binary as w;
        assert!(w(Some("application/octet-stream, application/json;q=0.5")));
        assert!(w(Some("application/octet-stream")));
        assert!(!w(Some("application/json")));
        assert!(!w(Some("application/json, application/octet-stream;q=0.5")));
        assert!(!w(Some("application/octet-stream;q=0")));
        assert!(!w(Some("*/*")));
        assert!(!w(None));
    }

    #[test]
    fn b64() {
        assert_eq!(super::base64(b"hello"), "aGVsbG8=");
        assert_eq!(super::base64(b"ab"), "YWI=");
        assert_eq!(super::base64(b""), "");
    }
}
