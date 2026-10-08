//! Model directory validation and model loading.
//!
//! The spec (section 4.2) says the model directory must contain exactly one
//! `*.gguf` file plus the matching `tokenizer.json`. We check that at startup
//! and fail with clear errors if anything is wrong, instead of letting
//! llama.cpp produce a confusing error deep inside inference later.

use std::path::{Path, PathBuf};

use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::model::LlamaModel;
use llama_cpp_2::model::params::LlamaModelParams;

/// Paths to the files inside the model directory, after validation.
#[derive(Debug)]
pub struct ModelDirPaths {
    /// The single `*.gguf` file found in the model directory.
    pub gguf: PathBuf,
    /// `tokenizer.json` from the model directory.
    pub tokenizer_json: PathBuf,
}

/// Check the model directory for exactly one `*.gguf` and a `tokenizer.json`.
///
/// Returns a plain `String` error because these are startup errors that get
/// printed once for the human reading the terminal — no need for a fancy
/// error type.
pub fn validate_model_dir(dir: &Path) -> Result<ModelDirPaths, String> {
    // The directory itself must exist.
    if !dir.is_dir() {
        return Err(format!("model dir not found: {}", dir.display()));
    }

    // Collect every *.gguf file in the directory.
    let mut ggufs: Vec<PathBuf> = Vec::new();
    for entry in std::fs::read_dir(dir).map_err(|e| format!("reading {}: {e}", dir.display()))? {
        let path = entry
            .map_err(|e| format!("reading {}: {e}", dir.display()))?
            .path();
        if path.is_file() && path.extension().is_some_and(|ext| ext == "gguf") {
            ggufs.push(path);
        }
    }

    // The spec requires exactly one GGUF, so we never have to guess which
    // file is "the" model.
    let gguf = match ggufs.as_slice() {
        [one] => one.clone(),
        [] => {
            return Err(format!(
                "no *.gguf file in {} — download the Gemma GGUF first (see spec section 4.2)",
                dir.display()
            ))
        }
        many => {
            let names: Vec<String> = many.iter().map(|p| p.display().to_string()).collect();
            return Err(format!(
                "expected exactly one *.gguf in {}, found {}:\n  {}",
                dir.display(),
                many.len(),
                names.join("\n  ")
            ));
        }
    };

    // tokenizer.json must sit next to the GGUF.
    let tokenizer_json = dir.join("tokenizer.json");
    if !tokenizer_json.is_file() {
        return Err(format!(
            "tokenizer.json not found in {} — it must be the Hugging Face tokenizer matching the GGUF",
            dir.display()
        ));
    }

    Ok(ModelDirPaths { gguf, tokenizer_json })
}

/// Load the model from disk, once, at startup.
///
/// `backend` proves llama.cpp is initialized; the model handle outlives this
/// function and is shared with all request handlers.
pub fn load_model(backend: &LlamaBackend, gguf: &Path) -> Result<LlamaModel, String> {
    println!("loading model from {} (this can take a while)...", gguf.display());

    // Default params, except for GPU offload: the spec says the app runs on
    // the **CPU only** (section 4.2), so we pin n_gpu_layers to 0 even on
    // machines where llama.cpp could offload to Metal/CUDA. The Q4
    // quantization is already baked into the GGUF.
    let params = LlamaModelParams::default().with_n_gpu_layers(0);

    LlamaModel::load_from_file(backend, gguf, &params)
        .map_err(|e| format!("failed to load {}: {e}", gguf.display()))
}
