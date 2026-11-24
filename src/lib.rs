//! inbed: a small, dependency-free text embedding server.

// `msg!` wraps objc_msgSend; its arguments are evaluated inside the unsafe call by design.
#![allow(clippy::macro_metavars_in_unsafe)]

pub mod backend;
pub mod bench;
pub mod cpu;
pub mod hub;
pub mod json;
pub mod model;
pub mod ring;
pub mod safetensors;
pub mod sys;
pub mod tokenizer;
pub mod unicode;
#[rustfmt::skip]
pub mod unicode_tables;
pub mod api;
#[cfg(target_os = "linux")]
pub mod cuda;
pub mod engine;
pub mod http;
#[cfg(target_os = "macos")]
pub mod metal;
