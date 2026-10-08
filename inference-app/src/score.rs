//! Scoring: turn a `POST /score` request into one logprob per text token.
//!
//! This is the heart of the inference app (spec 4.2, "Prompt construction"):
//!
//!   1. Apply the GGUF's own chat template to the prompt (generation prompt
//!      enabled), then append `assistant_prefix` verbatim — that's the context.
//!   2. Tokenize the context with BOS and special tokens parsed.
//!   3. Tokenize `text` separately with no BOS/EOS and special tokens parsed
//!      — these rules must match the frontend (spec 4.1). Because the two are
//!      tokenized separately, tokens never merge across the boundary, so the
//!      frontend's token count agrees with ours by construction.
//!   4. If the text token count != `expected_num_tokens`, return
//!      `UnexpectedNumTokens` without running any inference.
//!   5. One forward pass over context+text; for each text token t_i,
//!      logprob = log_softmax(logits at the previous position)[t_i], over the
//!      full vocabulary, raw logits, no sampling.

use crate::chat::ChatRenderer;
use llama_cpp_2::context::LlamaContext;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::LlamaModel;

/// Size of the KV cache / batch, in tokens (spec allows failing with a 500
/// "context too long" error, which happens naturally if a request exceeds
/// this). Hand-written puzzles stay far below it.
pub(crate) const N_CTX: u32 = 4096;

/// `POST /score` request body (spec 5.2).
#[derive(Debug, serde::Deserialize)]
pub struct ScoreRequest {
    /// The user message of the puzzle.
    pub prompt: String,
    /// Beginning of the assistant reply (may be "").
    pub assistant_prefix: String,
    /// The player's continuation.
    pub text: String,
    /// Token count from the puzzle; must be >= 1 (checked in the handler).
    pub expected_num_tokens: u32,
}

/// `POST /score` response body (spec 5.2). Serialized with serde's
/// internally-tagged representation, so JSON looks like
/// `{"type":"Logprobs","logprobs":[...]}` / `{"type":"UnexpectedNumTokens",...}`.
#[derive(Debug, serde::Serialize)]
#[serde(tag = "type")]
pub enum ScoreResponse {
    /// One logprob per text token, in order. Length == expected_num_tokens.
    Logprobs {
        logprobs: Vec<f64>,
    },
    /// The text tokenized to a different number of tokens than expected.
    /// Returned with HTTP 200; the frontend treats it like a wrong count.
    UnexpectedNumTokens {
        expected_num_tokens: u32,
        actual_num_tokens: u32,
    },
}

/// Score one request. The context (KV cache) must not be in use by anyone
/// else — the caller holds the mutex. `Err` means inference failure and maps
/// to HTTP 500 (spec 5.2 Errors).
pub fn score(
    model: &LlamaModel,
    renderer: &ChatRenderer,
    context: &mut LlamaContext,
    req: &ScoreRequest,
) -> Result<ScoreResponse, String> {
    // ---- 1. Build the context string (what the model sees) ----

    // One user message, with the generation prompt enabled, so the rendered
    // string ends exactly at the start of the model's turn. The template comes
    // from the GGUF itself (see chat.rs).
    let rendered = renderer.render_user_turn(model, &req.prompt)?;

    // The assistant prefix goes verbatim after the template's turn opener.
    let context_string = format!("{rendered}{}", req.assistant_prefix);

    // ---- 2. Tokenize ----

    let vocab = model.vocab();

    // Special tokens are parsed on both sides: the template contains literal
    // special-token strings like `<bos>` and `<|turn>`, and the player's text
    // may contain e.g. `<end_of_turn>`.
    //
    // BOS: the spec says the context is tokenized "with BOS added". Gemma's
    // chat template already starts with the literal `<bos>` special token, so
    // we tokenize with add_special=false and then add BOS ourselves if (and
    // only if) it is not already there. That gives exactly one BOS — what the
    // model sees in real inference — instead of the double BOS that a blind
    // add_special=true would produce here.
    let mut context_tokens = vocab.tokenize(context_string.as_bytes(), false, true);
    if context_tokens.first() != Some(&vocab.bos()) {
        context_tokens.insert(0, vocab.bos());
    }

    // The player's text: no BOS, no EOS, no dummy prefix — spec 4.1 rules.
    let text_tokens = vocab.tokenize(req.text.as_bytes(), false, true);

    // ---- 3. Token-count check (no inference if it fails) ----

    let actual = text_tokens.len() as u32;
    if actual != req.expected_num_tokens {
        return Ok(ScoreResponse::UnexpectedNumTokens {
            expected_num_tokens: req.expected_num_tokens,
            actual_num_tokens: actual,
        });
    }

    // ---- 4. One forward pass over context + text ----

    // Start from a clean KV cache so position 0 is really position 0.
    context.clear_kv_cache();

    let context_len = context_tokens.len();
    let text_len = text_tokens.len();
    let total = context_len + text_len;

    // The batch holds every token at its absolute position. Position i feeds
    // the model and produces the logits that predict the token at i+1, so we
    // request logits only at positions context_len-1 .. context_len+text_len-2
    // — exactly one row per text token.
    let mut batch = LlamaBatch::new(total, 1);
    for (i, token) in context_tokens.iter().chain(text_tokens.iter()).enumerate() {
        let want_logits = i + 1 >= context_len && i + 1 < total;
        batch
            .add(*token, i as i32, &[0], want_logits)
            .map_err(|e| format!("building the batch failed: {e}"))?;
    }

    // Decode errors (e.g. context too long for N_CTX) surface as a 500.
    context
        .decode(&mut batch)
        .map_err(|e| format!("inference failed (e.g. context too long): {e}"))?;

    // ---- 5. log_softmax(logits at the predicting position)[text token] ----

    // Text token k sits at batch position context_len + k and is predicted by
    // the logits at the position before it. get_logits_ith takes that *batch
    // position* (not a row number), so ask for context_len + k - 1.
    let mut logprobs = Vec::with_capacity(text_len);
    for (k, token) in text_tokens.iter().enumerate() {
        let row = context.get_logits_ith((context_len + k - 1) as i32);
        logprobs.push(log_softmax_pick(row, token.0 as usize));
    }

    Ok(ScoreResponse::Logprobs { logprobs })
}

/// `log_softmax(logits)[target]`, computed in one stable pass:
/// subtract the row max, sum the exps, then read off the target entry.
/// Raw logits, full vocabulary, temperature 1 — no sampling anywhere.
fn log_softmax_pick(logits: &[f32], target: usize) -> f64 {
    let max = logits
        .iter()
        .copied()
        .fold(f32::NEG_INFINITY, f32::max);

    let mut sum_exp = 0.0_f64;
    for &x in logits {
        sum_exp += (x as f64 - max as f64).exp();
    }

    logits[target] as f64 - max as f64 - sum_exp.ln()
}

#[cfg(test)]
mod tests {
    use super::*;

    // ---- Pure tests: log_softmax needs no model. ----

    #[test]
    fn log_softmax_matches_softmax() {
        // log_softmax over a known row, compared against a direct softmax.
        let row = [1.0_f32, 2.0, 3.0];
        let sum_e: f64 = row.iter().map(|x| (*x as f64).exp()).sum();
        for (i, _) in row.iter().enumerate() {
            // log(softmax) — the log of the probability, not the probability.
            let expected = ((row[i] as f64).exp() / sum_e).ln();
            let got = log_softmax_pick(&row, i);
            assert!((got - expected).abs() < 1e-9, "token {i}: {got} != {expected}");
        }
    }

    #[test]
    fn log_softmax_is_never_positive() {
        // A logprob is a log-probability, so it is always <= 0 (spec 7).
        let row = [0.0_f32, -5.5, 12.25, 1e-7, -1e-7];
        for i in 0..row.len() {
            assert!(log_softmax_pick(&row, i) <= 0.0);
        }
    }

    // ---- Model-backed tests (spec 7 "Server unit tests"). ----
    //
    // These load the real Gemma GGUF, so run them with
    //
    //     cd inference-app && cargo test --release
    //
    // (--release: the debug build of llama.cpp is very slow). The model
    // directory comes from CLANKERDLE_MODEL_DIR, defaulting to ../models
    // because cargo test runs from the crate directory.

    struct Engine {
        model: &'static LlamaModel,
        renderer: ChatRenderer,
        context: std::sync::Mutex<LlamaContext<'static>>,
    }

    /// Load the model once for the whole test binary and share it.
    /// The Box::leak trick gives the model a 'static lifetime, which the
    /// context's lifetime parameter needs; for a test binary (and for the
    /// server, see main.rs) that never exits any other way, "never dropped"
    /// is exactly the lifetime we want.
    fn engine() -> &'static Engine {
        static ENGINE: std::sync::OnceLock<Engine> = std::sync::OnceLock::new();
        ENGINE.get_or_init(|| {
            use llama_cpp_2::context::params::LlamaContextParams;
            use llama_cpp_2::llama_backend::LlamaBackend;
            use std::num::NonZeroU32;

            // CPU only — same reason as in main().
            std::env::set_var("GGML_METAL_DEVICES", "0");

            let model_dir =
                std::env::var("CLANKERDLE_MODEL_DIR").unwrap_or_else(|_| "../models".into());
            let paths = crate::model::validate_model_dir(model_dir.as_ref())
                .expect("model dir must be valid (download the GGUF first)");

            let backend: &'static LlamaBackend =
                Box::leak(Box::new(LlamaBackend::init().expect("backend init")));
            let model: &'static LlamaModel =
                Box::leak(Box::new(crate::model::load_model(backend, &paths.gguf).expect("model load")));

            let renderer = ChatRenderer::new(model).expect("chat renderer");

            let params = LlamaContextParams::default()
                .with_n_ctx(NonZeroU32::new(N_CTX))
                .with_n_batch(N_CTX);
            let context = model.new_context(backend, params).expect("context");

            Engine {
                model,
                renderer,
                context: std::sync::Mutex::new(context),
            }
        })
    }

    /// Run score() on a request, locking the shared context like the server does.
    fn run_score(req: &ScoreRequest) -> Result<ScoreResponse, String> {
        let engine = engine();
        let mut context = engine.context.lock().expect("context mutex poisoned");
        score(engine.model, &engine.renderer, &mut context, req)
    }

    fn test_request(prompt: &str, prefix: &str, text: &str, expected: u32) -> ScoreRequest {
        ScoreRequest {
            prompt: prompt.to_string(),
            assistant_prefix: prefix.to_string(),
            text: text.to_string(),
            expected_num_tokens: expected,
        }
    }

    #[test]
    fn wrong_token_count_is_unexpected_num_tokens() {
        // No inference runs, so this test is fast even in a debug build.
        let req = test_request("Hello!", "", "hello world", 999);
        match run_score(&req) {
            Ok(ScoreResponse::UnexpectedNumTokens {
                expected_num_tokens,
                actual_num_tokens,
            }) => {
                assert_eq!(expected_num_tokens, 999);
                assert_eq!(actual_num_tokens, 2, "\"hello world\" is 2 tokens");
            }
            other => panic!("expected UnexpectedNumTokens, got {other:?}"),
        }
    }

    #[test]
    fn logprobs_length_matches_and_are_not_positive() {
        // Pick any text and ask for its true token count, so this test is
        // self-consistent whatever the tokenizer does with it.
        let text = "I can help with that!";
        let engine = engine();
        let n = engine.model.vocab().tokenize(text.as_bytes(), false, true).len() as u32;
        assert!(n >= 1, "text should not be empty after tokenizing");

        let req = test_request("Can you help me?", "", text, n);
        match run_score(&req) {
            Ok(ScoreResponse::Logprobs { logprobs }) => {
                assert_eq!(logprobs.len(), n as usize, "one logprob per text token");
                for lp in &logprobs {
                    assert!(lp.is_finite(), "logprobs must be finite, got {lp}");
                    assert!(*lp <= 0.0, "logprobs must be <= 0, got {lp}");
                }
            }
            other => panic!("expected Logprobs, got {other:?}"),
        }
    }
}
