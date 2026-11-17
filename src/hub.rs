//! Resolve a model argument to a snapshot directory and a client-facing id.
//!
//! `--model` accepts:
//! * a registry name (`qwen/qwen3-embedding-0p6b`)
//! * a Hugging Face repo id that is in the local cache (`sentence-transformers/all-MiniLM-L6-v2`)
//! * a directory (`./models/minilm`)
//! * `<id>=<repo or dir>` to serve it under a chosen id

use std::path::{Path, PathBuf};

pub struct Known {
    /// client-facing id (turbopuffer's model name where one exists)
    pub id: &'static str,
    pub repo: &'static str,
    /// Matryoshka dims the model is served at; `None` = only its native dims
    pub allowed_dims: Option<&'static [usize]>,
}

pub static REGISTRY: &[Known] = &[
    Known {
        id: "qwen/qwen3-embedding-0p6b",
        repo: "Qwen/Qwen3-Embedding-0.6B",
        allowed_dims: Some(&[256, 384, 512, 768, 1024]),
    },
    Known {
        id: "sentence-transformers/all-MiniLM-L6-v2",
        repo: "sentence-transformers/all-MiniLM-L6-v2",
        allowed_dims: None,
    },
];

pub struct Resolved {
    pub id: String,
    pub dir: PathBuf,
    pub allowed_dims: Option<Vec<usize>>,
}

pub fn hf_cache() -> PathBuf {
    if let Ok(p) = std::env::var("HF_HUB_CACHE") {
        return PathBuf::from(p);
    }
    if let Ok(p) = std::env::var("HF_HOME") {
        return Path::new(&p).join("hub");
    }
    let home = std::env::var("HOME").unwrap_or_else(|_| ".".into());
    Path::new(&home).join(".cache/huggingface/hub")
}

/// `~/.cache/huggingface/hub/models--org--name/snapshots/<refs/main>`
pub fn cached_repo(repo: &str) -> Option<PathBuf> {
    let base = hf_cache().join(format!("models--{}", repo.replace('/', "--")));
    if let Ok(r) = std::fs::read_to_string(base.join("refs/main")) {
        let p = base.join("snapshots").join(r.trim());
        if p.is_dir() {
            return Some(p);
        }
    }
    // any snapshot with a config.json
    let mut snaps: Vec<PathBuf> = std::fs::read_dir(base.join("snapshots"))
        .ok()?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.join("config.json").exists())
        .collect();
    snaps.sort();
    snaps.pop()
}

pub fn resolve(arg: &str) -> Result<Resolved, String> {
    let (id, target) = match arg.split_once('=') {
        Some((id, t)) => (Some(id.to_string()), t),
        None => (None, arg),
    };
    let known = REGISTRY
        .iter()
        .find(|k| k.id.eq_ignore_ascii_case(target) || k.repo == target);
    let repo = known.map(|k| k.repo).unwrap_or(target);
    let dir = if Path::new(target).join("config.json").exists() {
        PathBuf::from(target)
    } else {
        cached_repo(repo).ok_or_else(|| format!("model {target:?} not found: not a directory and not in the Hugging Face cache ({}). Download it with `hf download {repo}`", hf_cache().display()))?
    };
    let id = id
        .or_else(|| known.map(|k| k.id.to_string()))
        .unwrap_or_else(|| {
            if Path::new(target).is_dir() {
                Path::new(target)
                    .file_name()
                    .map(|s| s.to_string_lossy().to_string())
                    .unwrap_or_else(|| target.to_string())
            } else {
                target.to_string()
            }
        });
    Ok(Resolved {
        id,
        dir,
        allowed_dims: known.and_then(|k| k.allowed_dims.map(|d| d.to_vec())),
    })
}
