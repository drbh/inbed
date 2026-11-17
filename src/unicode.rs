//! Unicode properties and normalization (NFD / NFC) over generated tables.

use crate::unicode_tables as t;

fn in_ranges(r: &[(u32, u32)], c: char) -> bool {
    let c = c as u32;
    r.binary_search_by(|&(a, b)| {
        if c < a {
            std::cmp::Ordering::Greater
        } else if c > b {
            std::cmp::Ordering::Less
        } else {
            std::cmp::Ordering::Equal
        }
    })
    .is_ok()
}

/// `\p{L}`
pub fn is_letter(c: char) -> bool {
    if c.is_ascii() {
        return c.is_ascii_alphabetic();
    }
    in_ranges(t::LETTER, c)
}

/// `\p{N}`
pub fn is_number(c: char) -> bool {
    if c.is_ascii() {
        return c.is_ascii_digit();
    }
    in_ranges(t::NUMBER, c)
}

/// General category P*.
pub fn is_punct_category(c: char) -> bool {
    in_ranges(t::PUNCT, c)
}

/// General category Mn.
pub fn is_nonspacing_mark(c: char) -> bool {
    !c.is_ascii() && in_ranges(t::NONSPACING_MARK, c)
}

/// Cc | Cf | Co | Cn.
pub fn is_other(c: char) -> bool {
    in_ranges(t::OTHER, c)
}

pub fn ccc(c: char) -> u8 {
    let c = c as u32;
    if c < 0x300 {
        return 0;
    }
    match t::CCC.binary_search_by_key(&c, |&(a, _)| a) {
        Ok(i) => t::CCC[i].1,
        Err(_) => 0,
    }
}

const S_BASE: u32 = 0xAC00;
const L_BASE: u32 = 0x1100;
const V_BASE: u32 = 0x1161;
const T_BASE: u32 = 0x11A7;
const L_COUNT: u32 = 19;
const V_COUNT: u32 = 21;
const T_COUNT: u32 = 28;
const N_COUNT: u32 = V_COUNT * T_COUNT;
const S_COUNT: u32 = L_COUNT * N_COUNT;

fn decompose_into(c: char, out: &mut Vec<char>) {
    let cp = c as u32;
    if (S_BASE..S_BASE + S_COUNT).contains(&cp) {
        let s = cp - S_BASE;
        out.push(char::from_u32(L_BASE + s / N_COUNT).unwrap());
        out.push(char::from_u32(V_BASE + (s % N_COUNT) / T_COUNT).unwrap());
        let tt = s % T_COUNT;
        if tt != 0 {
            out.push(char::from_u32(T_BASE + tt).unwrap());
        }
        return;
    }
    if cp >= 0xC0
        && let Ok(i) = t::DECOMP_INDEX.binary_search_by_key(&cp, |&(a, _, _)| a)
    {
        let (_, off, n) = t::DECOMP_INDEX[i];
        for &d in &t::DECOMP_DATA[off as usize..off as usize + n as usize] {
            decompose_into(char::from_u32(d).unwrap(), out);
        }
        return;
    }
    out.push(c);
}

/// Canonical decomposition with canonical ordering.
pub fn nfd_chars(s: &str) -> Vec<char> {
    let mut out = Vec::with_capacity(s.len());
    for c in s.chars() {
        decompose_into(c, &mut out);
    }
    // canonical ordering: stable sort runs of non-starters by ccc
    let mut i = 0;
    while i < out.len() {
        if ccc(out[i]) == 0 {
            i += 1;
            continue;
        }
        let start = i;
        while i < out.len() && ccc(out[i]) != 0 {
            i += 1;
        }
        out[start..i].sort_by_key(|&c| ccc(c));
    }
    out
}

pub fn nfd(s: &str) -> String {
    nfd_chars(s).into_iter().collect()
}

fn compose_pair(a: char, b: char) -> Option<char> {
    let (a, b) = (a as u32, b as u32);
    // Hangul LV / LVT
    if (L_BASE..L_BASE + L_COUNT).contains(&a) && (V_BASE..V_BASE + V_COUNT).contains(&b) {
        return char::from_u32(S_BASE + ((a - L_BASE) * V_COUNT + (b - V_BASE)) * T_COUNT);
    }
    if (S_BASE..S_BASE + S_COUNT).contains(&a)
        && (a - S_BASE).is_multiple_of(T_COUNT)
        && (T_BASE + 1..T_BASE + T_COUNT).contains(&b)
    {
        return char::from_u32(a + (b - T_BASE));
    }
    t::COMPOSE
        .binary_search_by(|&(x, y, _)| (x, y).cmp(&(a, b)))
        .ok()
        .map(|i| char::from_u32(t::COMPOSE[i].2).unwrap())
}

pub fn nfc(s: &str) -> String {
    if s.is_ascii() {
        return s.to_string();
    }
    let d = nfd_chars(s);
    let mut out: Vec<char> = Vec::with_capacity(d.len());
    let mut starter: Option<usize> = None;
    let mut prev_cc: u8 = 0;
    for c in d {
        let cc = ccc(c);
        if let Some(si) = starter {
            let adjacent = si + 1 == out.len();
            if (adjacent || (prev_cc != 0 && prev_cc < cc))
                && let Some(p) = compose_pair(out[si], c)
            {
                out[si] = p;
                continue;
            }
        }
        if cc == 0 {
            starter = Some(out.len());
        }
        prev_cc = cc;
        out.push(c);
    }
    out.into_iter().collect()
}

/// BERT's "CJK unified ideograph" ranges.
pub fn is_chinese_char(c: char) -> bool {
    matches!(c as u32,
        0x4E00..=0x9FFF | 0x3400..=0x4DBF | 0x20000..=0x2A6DF | 0x2A700..=0x2B73F |
        0x2B740..=0x2B81F | 0x2B820..=0x2CEAF | 0xF900..=0xFAFF | 0x2F800..=0x2FA1F)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn normalization() {
        assert_eq!(nfd("é"), "e\u{301}");
        assert_eq!(nfc("e\u{301}"), "é");
        assert_eq!(nfc("한국어"), "한국어");
        assert_eq!(nfd("한").chars().count(), 3);
        assert_eq!(nfc(&nfd("한국어")), "한국어");
        assert_eq!(nfc("a\u{323}\u{302}"), "\u{1EAD}"); // ậ via reordering
        assert_eq!(nfc("Ångström"), "Ångström");
        assert!(is_letter('é') && is_letter('中') && !is_letter('1') && is_number('٣'));
        assert!(is_punct_category('¿') && is_other('\u{200b}'));
    }
}
