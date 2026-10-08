//! Parity-test helper: tokenize a corpus of strings with the GGUF's own
//! tokenizer and print the token IDs as JSON.
//!
//! This is the "server side" of the tokenizer parity test (spec section 7):
//! the Node script `frontend/scripts/parity.mjs` runs this bin, runs the
//! *frontend* tokenizer on the same corpus, and compares the token IDs.
//!
//! The tokenization rules come from `inference_app::score::tokenize_text` —
//! the very same function `POST /score` scores with — so the rules cannot
//! drift between the server and this helper.
//!
//! Usage (from the repo root):
//!
//!     cd inference-app
//!     cargo run --release --bin tokenize -- ../frontend/tests/parity-corpus.json
//!
//! The model dir comes from CLANKERDLE_MODEL_DIR and defaults to ../models
//! (cargo runs bins from the crate directory).
//!
//! The model is loaded with `vocab_only`, which skips the weights entirely:
//! startup takes seconds instead of a minute, and the vocab is all that
//! tokenization needs.

use std::path::Path;

use inference_app::model;
use inference_app::score::tokenize_text;
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::model::LlamaModel;
use llama_cpp_2::model::params::LlamaModelParams;

/// The corpus file: `{ "cases": [ "...", ... ] }`.
#[derive(serde::Deserialize)]
struct Corpus {
    cases: Vec<String>,
}

fn main() {
    // CPU only (spec 4.2) — and it also skips the Metal cleanup crash some
    // macOS versions hit at exit. Same as main().
    std::env::set_var("GGML_METAL_DEVICES", "0");

    let corpus_path = match std::env::args().nth(1) {
        Some(path) => path,
        None => {
            eprintln!("usage: cargo run --release --bin tokenize -- <parity-corpus.json>");
            std::process::exit(2);
        }
    };

    let corpus: Corpus = serde_json::from_str(
        &std::fs::read_to_string(&corpus_path).unwrap_or_else(|e| {
            eprintln!("error: reading {corpus_path}: {e}");
            std::process::exit(1);
        }),
    )
    .unwrap_or_else(|e| {
        eprintln!("error: parsing {corpus_path}: {e}");
        std::process::exit(1);
    });

    let model_dir =
        std::env::var("CLANKERDLE_MODEL_DIR").unwrap_or_else(|_| "../models".into());
    let paths = model::validate_model_dir(Path::new(&model_dir)).unwrap_or_else(|e| {
        eprintln!("error: {e}");
        std::process::exit(1);
    });

    // vocab_only + CPU: no weights get loaded or offloaded anywhere.
    let backend = LlamaBackend::init().unwrap_or_else(|e| {
        eprintln!("error: failed to initialize llama backend: {e}");
        std::process::exit(1);
    });
    let params = LlamaModelParams::default()
        .with_vocab_only(true)
        .with_n_gpu_layers(0);
    let model = LlamaModel::load_from_file(&backend, &paths.gguf, &params).unwrap_or_else(|e| {
        eprintln!("error: failed to load {}: {e}", paths.gguf.display());
        std::process::exit(1);
    });
    eprintln!(
        "tokenizing {} cases with {} ({} vocab entries)",
        corpus.cases.len(),
        paths.gguf.display(),
        model.n_vocab()
    );

    // One output entry per corpus case: a JSON array of {id, text} per token.
    // `text` is the raw token spelling (lossy for byte-fallback tokens) and
    // exists purely so parity mismatches are readable in the test output.
    let mut output = Vec::with_capacity(corpus.cases.len());
    for case in &corpus.cases {
        let tokens: Vec<serde_json::Value> = tokenize_text(&model, case)
            .iter()
            .map(|&token| {
                let text = model
                    .vocab()
                    .text(token)
                    .map(|raw| raw.to_string_lossy().to_string())
                    .unwrap_or_default();
                serde_json::json!({ "id": token.0, "text": text })
            })
            .collect();
        output.push(tokens);
    }

    // Machine-readable output on stdout (the parity script reads this);
    // progress notes go to stderr so they don't pollute the pipe.
    println!("{}", serde_json::to_string(&output).unwrap());
}