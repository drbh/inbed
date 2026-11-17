//! Tokenizers loaded from Hugging Face `tokenizer.json`: BERT WordPiece and byte-level BPE (GPT-2 / Qwen).

use crate::json::Json;
use crate::unicode;
use std::collections::HashMap;
use std::path::Path;

pub struct Tokenizer {
    kind: Kind,
    /// special ids around the sequence (TemplateProcessing "single")
    prefix: Vec<u32>,
    suffix: Vec<u32>,
    /// added tokens matched in raw text before normalization: (content, id)
    added: Vec<(String, u32)>,
}

enum Kind {
    WordPiece(WordPiece),
    Bpe(Box<Bpe>),
}

struct WordPiece {
    vocab: HashMap<String, u32>,
    unk: u32,
    prefix: String,
    max_chars: usize,
    lowercase: bool,
    strip_accents: bool,
    clean_text: bool,
    chinese_chars: bool,
}

struct Bpe {
    vocab: HashMap<String, u32>,
    /// (left id, right id) → (rank, merged id)
    merges: HashMap<(u32, u32), (u32, u32)>,
    byte_ids: [u32; 256],
    nfc: bool,
}

impl Tokenizer {
    pub fn from_file(path: &Path) -> Result<Tokenizer, String> {
        let s = std::fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
        Self::from_json(&Json::parse(&s)?)
    }

    pub fn from_json(j: &Json) -> Result<Tokenizer, String> {
        let model = j.get("model").ok_or("tokenizer.json: no model")?;
        let mut vocab: HashMap<String, u32> = HashMap::new();
        for (k, v) in model
            .get("vocab")
            .and_then(Json::as_obj)
            .ok_or("tokenizer.json: no vocab")?
        {
            vocab.insert(k.clone(), v.as_u64().ok_or("bad vocab id")? as u32);
        }
        let mut added = Vec::new();
        for a in j.get("added_tokens").and_then(Json::as_arr).unwrap_or(&[]) {
            let (Some(c), Some(id)) = (
                a.get("content").and_then(Json::as_str),
                a.get("id").and_then(Json::as_u64),
            ) else {
                continue;
            };
            vocab.entry(c.to_string()).or_insert(id as u32);
            added.push((c.to_string(), id as u32));
        }
        // longest first so overlapping specials match greedily
        added.sort_by(|a, b| b.0.len().cmp(&a.0.len()));

        let (mut prefix, mut suffix) = (Vec::new(), Vec::new());
        let mut templates = vec![];
        if let Some(pp) = j.get("post_processor") {
            match pp.get("type").and_then(Json::as_str) {
                Some("Sequence") => templates.extend(
                    pp.get("processors")
                        .and_then(Json::as_arr)
                        .unwrap_or(&[])
                        .iter(),
                ),
                _ => templates.push(pp),
            }
        }
        for t in templates {
            match t.get("type").and_then(Json::as_str) {
                Some("TemplateProcessing") => {
                    let specials = t.get("special_tokens");
                    let mut seen_seq = false;
                    for piece in t.get("single").and_then(Json::as_arr).unwrap_or(&[]) {
                        if piece.get("Sequence").is_some() {
                            seen_seq = true;
                        } else if let Some(sp) = piece.get("SpecialToken") {
                            let id = sp.get("id").and_then(Json::as_str).unwrap_or("");
                            let ids = specials
                                .and_then(|s| s.get(id))
                                .and_then(|s| s.get("ids"))
                                .and_then(Json::as_arr)
                                .ok_or_else(|| format!("template special {id} has no ids"))?;
                            let ids = ids.iter().filter_map(Json::as_u64).map(|x| x as u32);
                            if seen_seq {
                                suffix.extend(ids)
                            } else {
                                prefix.extend(ids)
                            }
                        }
                    }
                }
                Some("BertProcessing") => {
                    prefix.push(
                        t.get("cls")
                            .and_then(Json::as_arr)
                            .and_then(|a| a.get(1))
                            .and_then(Json::as_u64)
                            .ok_or("bad cls")? as u32,
                    );
                    suffix.push(
                        t.get("sep")
                            .and_then(Json::as_arr)
                            .and_then(|a| a.get(1))
                            .and_then(Json::as_u64)
                            .ok_or("bad sep")? as u32,
                    );
                }
                _ => {}
            }
        }

        let norm = j.get("normalizer").unwrap_or(&Json::Null);
        let kind = match model.get("type").and_then(Json::as_str) {
            Some("WordPiece") => {
                let lowercase = norm
                    .get("lowercase")
                    .and_then(Json::as_bool)
                    .unwrap_or(true);
                let strip_accents = norm
                    .get("strip_accents")
                    .and_then(Json::as_bool)
                    .unwrap_or(lowercase);
                let unk_tok = model
                    .get("unk_token")
                    .and_then(Json::as_str)
                    .unwrap_or("[UNK]");
                Kind::WordPiece(WordPiece {
                    unk: *vocab.get(unk_tok).ok_or("unk token not in vocab")?,
                    prefix: model
                        .get("continuing_subword_prefix")
                        .and_then(Json::as_str)
                        .unwrap_or("##")
                        .to_string(),
                    max_chars: model
                        .get("max_input_chars_per_word")
                        .and_then(Json::as_u64)
                        .unwrap_or(100) as usize,
                    lowercase,
                    strip_accents,
                    clean_text: norm
                        .get("clean_text")
                        .and_then(Json::as_bool)
                        .unwrap_or(true),
                    chinese_chars: norm
                        .get("handle_chinese_chars")
                        .and_then(Json::as_bool)
                        .unwrap_or(true),
                    vocab,
                })
            }
            Some("BPE") => {
                let mut merges = HashMap::new();
                for (rank, m) in model
                    .get("merges")
                    .and_then(Json::as_arr)
                    .ok_or("BPE: no merges")?
                    .iter()
                    .enumerate()
                {
                    let (a, b) = match m {
                        Json::Str(s) => {
                            let mut it = s.splitn(2, ' ');
                            (
                                it.next().unwrap_or("").to_string(),
                                it.next().unwrap_or("").to_string(),
                            )
                        }
                        Json::Arr(p) if p.len() == 2 => (
                            p[0].as_str().unwrap_or("").to_string(),
                            p[1].as_str().unwrap_or("").to_string(),
                        ),
                        _ => return Err("BPE: bad merge".into()),
                    };
                    let (Some(&ia), Some(&ib), Some(&im)) =
                        (vocab.get(&a), vocab.get(&b), vocab.get(&format!("{a}{b}")))
                    else {
                        continue;
                    };
                    merges.entry((ia, ib)).or_insert((rank as u32, im));
                }
                let b2u = bytes_to_unicode();
                let mut byte_ids = [0u32; 256];
                for (i, c) in b2u.iter().enumerate() {
                    byte_ids[i] = *vocab
                        .get(&c.to_string())
                        .ok_or("BPE: byte symbol missing from vocab")?;
                }
                let nfc = has_normalizer(norm, "NFC");
                if let Some(pt) = j.get("pre_tokenizer") {
                    check_qwen_pretokenizer(pt)?;
                }
                Kind::Bpe(Box::new(Bpe {
                    vocab,
                    merges,
                    byte_ids,
                    nfc,
                }))
            }
            t => return Err(format!("unsupported tokenizer model {t:?}")),
        };
        Ok(Tokenizer {
            kind,
            prefix,
            suffix,
            added,
        })
    }

    pub fn specials_len(&self) -> usize {
        self.prefix.len() + self.suffix.len()
    }

    /// Token ids with specials, truncated so the result has at most `max_len` ids.
    pub fn encode(&self, text: &str, max_len: usize) -> Vec<u32> {
        let budget = max_len.saturating_sub(self.specials_len());
        let mut body = Vec::new();
        for (seg, special) in self.split_added(text) {
            if body.len() >= budget {
                break;
            }
            match special {
                Some(id) => body.push(id),
                None => match &self.kind {
                    Kind::WordPiece(w) => w.encode(seg, &mut body),
                    Kind::Bpe(b) => b.encode(seg, &mut body),
                },
            }
        }
        body.truncate(budget);
        let mut out = Vec::with_capacity(body.len() + self.specials_len());
        out.extend_from_slice(&self.prefix);
        out.extend(body);
        out.extend_from_slice(&self.suffix);
        out
    }

    fn split_added<'a>(&self, text: &'a str) -> Vec<(&'a str, Option<u32>)> {
        let mut out = Vec::new();
        if self.added.is_empty() {
            out.push((text, None));
            return out;
        }
        let mut last = 0;
        let mut i = 0;
        let b = text.as_bytes();
        while i < b.len() {
            if (b[i] == b'<' || b[i] == b'[')
                && let Some((c, id)) = self
                    .added
                    .iter()
                    .find(|(c, _)| text[i..].starts_with(c.as_str()))
            {
                if last < i {
                    out.push((&text[last..i], None));
                }
                out.push((&text[i..i + c.len()], Some(*id)));
                i += c.len();
                last = i;
                continue;
            }
            i += 1;
        }
        if last < text.len() {
            out.push((&text[last..], None));
        }
        out
    }
}

fn has_normalizer(n: &Json, ty: &str) -> bool {
    match n.get("type").and_then(Json::as_str) {
        Some("Sequence") => n
            .get("normalizers")
            .and_then(Json::as_arr)
            .unwrap_or(&[])
            .iter()
            .any(|x| has_normalizer(x, ty)),
        Some(t) => t == ty,
        None => false,
    }
}

const QWEN_PATTERN: &str = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";

/// The split pattern is hand-compiled below; refuse tokenizers that use a different one.
fn check_qwen_pretokenizer(pt: &Json) -> Result<(), String> {
    let mut found = false;
    let list: Vec<&Json> = match pt.get("type").and_then(Json::as_str) {
        Some("Sequence") => pt
            .get("pretokenizers")
            .and_then(Json::as_arr)
            .unwrap_or(&[])
            .iter()
            .collect(),
        _ => vec![pt],
    };
    for p in list {
        match p.get("type").and_then(Json::as_str) {
            Some("Split") => {
                let pat = p
                    .get("pattern")
                    .and_then(|x| x.get("Regex"))
                    .and_then(Json::as_str)
                    .unwrap_or("");
                if pat != QWEN_PATTERN {
                    return Err(format!("unsupported BPE split pattern {pat:?}"));
                }
                found = true;
            }
            Some("ByteLevel") => {
                if p.get("use_regex").and_then(Json::as_bool) == Some(true) && !found {
                    return Err(
                        "ByteLevel pre-tokenizer with its own regex is not supported".into(),
                    );
                }
            }
            t => return Err(format!("unsupported pre-tokenizer {t:?}")),
        }
    }
    if !found {
        return Err("BPE tokenizer without a Split pre-tokenizer is not supported".into());
    }
    Ok(())
}

// ------------------------------------------------------------------ WordPiece

impl WordPiece {
    fn normalize(&self, s: &str) -> String {
        let mut out = String::with_capacity(s.len() + 8);
        for c in s.chars() {
            if self.clean_text {
                if c == '\0'
                    || c == '\u{fffd}'
                    || (!matches!(c, '\t' | '\n' | '\r') && unicode::is_other(c))
                {
                    continue;
                }
                if matches!(c, '\t' | '\n' | '\r') || c.is_whitespace() {
                    out.push(' ');
                    continue;
                }
            }
            if self.chinese_chars && unicode::is_chinese_char(c) {
                out.push(' ');
                out.push(c);
                out.push(' ');
                continue;
            }
            out.push(c);
        }
        if self.strip_accents {
            out = unicode::nfd_chars(&out)
                .into_iter()
                .filter(|&c| !unicode::is_nonspacing_mark(c))
                .collect();
        }
        if self.lowercase {
            out = out.to_lowercase();
        }
        out
    }

    fn encode(&self, text: &str, out: &mut Vec<u32>) {
        let norm = self.normalize(text);
        let mut word = String::new();
        let flush = |w: &mut String, out: &mut Vec<u32>| {
            if !w.is_empty() {
                self.word(w, out);
                w.clear();
            }
        };
        for c in norm.chars() {
            if c.is_whitespace() {
                flush(&mut word, out);
            } else if c.is_ascii_punctuation() || unicode::is_punct_category(c) {
                flush(&mut word, out);
                word.push(c);
                flush(&mut word, out);
            } else {
                word.push(c);
            }
        }
        flush(&mut word, out);
    }

    fn word(&self, w: &str, out: &mut Vec<u32>) {
        let chars: Vec<(usize, char)> = w.char_indices().collect();
        if chars.len() > self.max_chars {
            out.push(self.unk);
            return;
        }
        let mut pieces = Vec::new();
        let mut start = 0;
        let mut buf = String::new();
        while start < chars.len() {
            let mut end = chars.len();
            let mut found = None;
            while start < end {
                let a = chars[start].0;
                let b = if end == chars.len() {
                    w.len()
                } else {
                    chars[end].0
                };
                buf.clear();
                if start > 0 {
                    buf.push_str(&self.prefix);
                }
                buf.push_str(&w[a..b]);
                if let Some(&id) = self.vocab.get(&buf) {
                    found = Some(id);
                    break;
                }
                end -= 1;
            }
            match found {
                Some(id) => {
                    pieces.push(id);
                    start = end;
                }
                None => {
                    out.push(self.unk);
                    return;
                }
            }
        }
        out.extend(pieces);
    }
}

// ------------------------------------------------------------------ byte-level BPE

fn bytes_to_unicode() -> Vec<char> {
    let mut bs: Vec<u32> = (b'!' as u32..=b'~' as u32)
        .chain(0xA1..=0xAC)
        .chain(0xAE..=0xFF)
        .collect();
    let mut cs = bs.clone();
    let mut n = 0;
    for b in 0..256u32 {
        if !bs.contains(&b) {
            bs.push(b);
            cs.push(256 + n);
            n += 1;
        }
    }
    let mut out = vec!['\0'; 256];
    for (b, c) in bs.iter().zip(cs) {
        out[*b as usize] = char::from_u32(c).unwrap();
    }
    out
}

fn is_ws(c: char) -> bool {
    c.is_whitespace()
}
fn is_ln(c: char) -> bool {
    unicode::is_letter(c) || unicode::is_number(c)
}

/// Qwen2 / GPT-4-style split, hand-compiled from `QWEN_PATTERN`. Returns piece end offsets in chars.
pub fn qwen_split(cs: &[char]) -> Vec<(usize, usize)> {
    let n = cs.len();
    let mut out = Vec::new();
    let mut i = 0;
    while i < n {
        let end = qwen_match(cs, i);
        out.push((i, end));
        i = end;
    }
    out
}

fn qwen_match(cs: &[char], i: usize) -> usize {
    let n = cs.len();
    let c = cs[i];
    // 1. (?i:'s|'t|'re|'ve|'m|'ll|'d)
    if c == '\'' && i + 1 < n {
        let a = cs[i + 1].to_ascii_lowercase();
        let b = if i + 2 < n {
            cs[i + 2].to_ascii_lowercase()
        } else {
            '\0'
        };
        if matches!(a, 's' | 't' | 'm' | 'd') {
            return i + 2;
        }
        if matches!((a, b), ('r', 'e') | ('v', 'e') | ('l', 'l')) {
            return i + 3;
        }
    }
    // 2. [^\r\n\p{L}\p{N}]?\p{L}+
    {
        let mut j = i;
        if !matches!(c, '\r' | '\n') && !is_ln(c) {
            j += 1;
        }
        let s = j;
        while j < n && unicode::is_letter(cs[j]) {
            j += 1;
        }
        if j > s {
            return j;
        }
        // the optional char might itself have been the only candidate; retry without it
        if s != i {
            let mut j = i;
            while j < n && unicode::is_letter(cs[j]) {
                j += 1;
            }
            if j > i {
                return j;
            }
        }
    }
    // 3. \p{N}
    if unicode::is_number(c) {
        return i + 1;
    }
    // 4.  ?[^\s\p{L}\p{N}]+[\r\n]*
    {
        let mut j = i;
        if c == ' ' {
            j += 1;
        }
        let s = j;
        while j < n && !is_ws(cs[j]) && !is_ln(cs[j]) {
            j += 1;
        }
        if j > s {
            while j < n && matches!(cs[j], '\r' | '\n') {
                j += 1;
            }
            return j;
        }
    }
    // 5. \s*[\r\n]+
    let mut j = i;
    while j < n && is_ws(cs[j]) {
        j += 1;
    }
    if let Some(k) = (i..j).rev().find(|&k| matches!(cs[k], '\r' | '\n')) {
        return k + 1;
    }
    // 6. \s+(?!\S)
    if j > i {
        if j == n {
            return j;
        }
        if j - i >= 2 {
            return j - 1;
        }
        // 7. \s+
        return j;
    }
    // unreachable for valid input, but always make progress
    i + 1
}

impl Bpe {
    fn encode(&self, text: &str, out: &mut Vec<u32>) {
        let norm;
        let text = if self.nfc {
            norm = unicode::nfc(text);
            norm.as_str()
        } else {
            text
        };
        let cs: Vec<char> = text.chars().collect();
        let mut buf = [0u8; 4];
        let mut bytes = Vec::new();
        for (a, b) in qwen_split(&cs) {
            bytes.clear();
            for &c in &cs[a..b] {
                bytes.extend_from_slice(c.encode_utf8(&mut buf).as_bytes());
            }
            self.bpe(&bytes, out);
        }
    }

    fn bpe(&self, bytes: &[u8], out: &mut Vec<u32>) {
        let mut syms: Vec<u32> = bytes.iter().map(|&b| self.byte_ids[b as usize]).collect();
        // fast path: whole piece in vocab is common for short words, but BPE must still follow merges;
        // the merge loop below is exact.
        loop {
            let mut best: Option<(u32, usize, u32)> = None;
            for k in 0..syms.len().saturating_sub(1) {
                if let Some(&(rank, id)) = self.merges.get(&(syms[k], syms[k + 1]))
                    && best.is_none_or(|(r, _, _)| rank < r)
                {
                    best = Some((rank, k, id));
                }
            }
            let Some((rank, _, _)) = best else { break };
            // merge every occurrence of this pair, left to right
            let mut k = 0;
            let mut next = Vec::with_capacity(syms.len());
            while k < syms.len() {
                if k + 1 < syms.len()
                    && let Some(&(r, id)) = self.merges.get(&(syms[k], syms[k + 1]))
                    && r == rank
                {
                    next.push(id);
                    k += 2;
                    continue;
                }
                next.push(syms[k]);
                k += 1;
            }
            syms = next;
        }
        out.extend(syms);
        let _ = &self.vocab;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn split(s: &str) -> Vec<String> {
        let cs: Vec<char> = s.chars().collect();
        qwen_split(&cs)
            .into_iter()
            .map(|(a, b)| cs[a..b].iter().collect())
            .collect()
    }
    #[test]
    fn qwen_pattern() {
        assert_eq!(split("Hello, world!"), ["Hello", ",", " world", "!"]);
        assert_eq!(
            split("don't  stop\n\n  x"),
            ["don", "'t", " ", " stop", "\n\n", " ", " x"]
        );
        assert_eq!(split("a   "), ["a", "   "]);
        assert_eq!(split("123 abc"), ["1", "2", "3", " abc"]);
        assert_eq!(split(" ...\n"), [" ...\n"]);
    }
}
