//! Clankerdle inference app — core logic, no HTTP.
//!
//! The library crate holds everything that is shared between the HTTP server
//! (`main.rs`) and the tooling (`bin/tokenize.rs`, the parity-test helper):
//!
//!   - `model`  — model dir validation + loading (spec 4.2)
//!   - `chat`    — rendering the GGUF's own chat template
//!   - `score`   — the `POST /score` core path + the text tokenization rules
//!
//! Keeping the logic out of `main.rs` means the parity helper tokenizes with
//! the *exact same code* the server scores with — the rules cannot drift.

pub mod chat;
pub mod model;
pub mod score;