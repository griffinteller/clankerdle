# Clankerdle — Implementation Plan

Derived from `spec/clankerdle-spec.md`. The spec is the source of truth; this plan
is ordered so the highest-risk items (tokenizer parity, logit positioning) land early.

## Phase 0 — Scaffolding & assets

**0.1 Fetch model assets (prerequisite, no code)**
- Download Gemma 4 E2B Q4 GGUF + matching `tokenizer.json` into `models/` (gitignored).
- The GGUF must embed Gemma's chat template (the spec applies the GGUF's own template).

**0.2 Inference app crate**
- `cargo init inference-app` with deps: `actix-web`, `actix-cors`, `serde`,
  `serde_json`, `llama-cpp-2`, plus a tokenizer solution (see risk item 2).
- `PORT` (default 8080, bound to 127.0.0.1) and `CLANKERDLE_MODEL_DIR`
  (default `./models`) config.
- Startup: validate exactly one `*.gguf` + `tokenizer.json` exist, fail with clear
  errors otherwise; load model once.

**0.3 Frontend**
- `npm create vite@latest frontend -- --template react-ts`.
- Add `@huggingface/transformers` for `PreTrainedTokenizer` from the fetched
  `tokenizer.json`.
- `VITE_INFERENCE_URL` env (default `http://localhost:8080`); permissive CORS on
  the server side.

## Phase 1 — Inference app

Ordered so the risky parts land early.

**1.1 `GET /tokenizer`** — stream `tokenizer.json` from the model dir,
`application/json`, 200.

**1.2 `POST /score` core path**
1. Parse `ScoreRequest`; 400 on malformed JSON or `expected_num_tokens < 1`.
2. Build context: apply the GGUF's chat template to `prompt` with the generation
   prompt enabled, then append `assistant_prefix` verbatim.
3. Tokenize context **with BOS**, special tokens parsed; tokenize `text`
   **separately, no BOS/EOS**, special tokens parsed.
4. If text token count ≠ `expected_num_tokens` → `UnexpectedNumTokens` variant
   (no inference).
5. One forward pass over context+text; `logprob = log_softmax(logits at position
   i−1 over full vocab)[t_i]`, raw logits, no sampling/top-k/top-p.
6. Response: serde internally-tagged enum (`type: "Logprobs"` /
   `"UnexpectedNumTokens"`); 500 with `{ "error": string }` on inference failure.

**1.3 Server-side plumbing**
- Single `LlamaModel` + one context behind a mutex (requests scored serially).
- Permissive CORS (any origin; `GET`, `POST`, `Content-Type`).

**1.4 Server unit tests** (spec §7)
- `UnexpectedNumTokens` on count mismatch.
- Logprobs length == `expected_num_tokens`.
- All logprobs ≤ 0.
- (Manual later: logprobs for an "obvious" continuation near 0.)

## Phase 2 — Frontend tokenizer (parity-critical)

**2.1 Tokenizer loading** — fetch from `GET /tokenizer`, build
`PreTrainedTokenizer`; `loading…` while in flight, plain error message on failure.

**2.2 Client-side tokenization of player input**
- Tokenize only the input; **no** BOS/EOS, no dummy prefix; special-token
  strings (e.g. `<end_of_turn>`) recognized as single special tokens.
- Render each token by decoding its ID individually, so spans concatenate back
  exactly.

**2.3 Tokenizer parity test (important!)**
- Fixed string corpus: leading/trailing spaces, newlines, emoji/non-ASCII,
  `<end_of_turn>`, whitespace-only.
- Node script (or Vite test) comparing frontend vs. server tokenization —
  counts and IDs. This gates everything else; per spec the parity holds "by
  construction," so the test validates our construction rules, and
  `UnexpectedNumTokens` is the safety net.

## Phase 3 — Game UI (React + TS, monospace on plain background)

**3.1 Puzzle data** — `frontend/src/puzzles.json`:
`{id, prompt, assistant_prefix, expected_num_tokens}[]`, bundled at build time;
server never sees puzzles.

**3.2 App state machine** — `playing` → `scoring` (input locked) → `revealed` →
next challenge / `summary`. No persistence; refresh restarts at challenge 1.
One submission per challenge.

**3.3 Challenge screen**
- Line 1: `User: <prompt>`; Line 2: `Assistant: <assistant_prefix>` + live
  input + blinking cursor.
- Input supports free typing, arrows, backspace, paste; Shift+Enter inserts a
  newline; Enter submits; any text allowed (incl. whitespace-only).

**3.4 Live tokenization render**
- Re-tokenize on every change; cycle token background colors from a small pale
  palette (token 1 → A, token 2 → B, …).
- Whitespace-only tokens visible: space shows as a colored space, newline shows
  `↵` glyph before the break.
- Counter `n/N` (current count / `expected_num_tokens`).

**3.5 Scoring interactions**
- Enter with `n ≠ N`: ~300 ms horizontal shake, input stays editable. Same for
  `UnexpectedNumTokens` responses.
- Enter with `n = N`: lock input, `POST /score`. On 4xx/5xx: plain error line,
  keep input editable.

**3.6 Reveal state**
- One bar per token, aligned under the token's span, hanging from a common
  baseline; length ∝ `|logprob|`, clamped at 15 (≤ −15 → full-length bar);
  printed number is the true value, 2 decimal places.
- Challenge score = sum of logprobs, 2 decimal places.
- `press enter to continue...`; Enter → next challenge or summary.
- Never show the model's preferred completion or alternatives.

**3.7 Summary screen** — rows of `#idx  score  emoji-squares` (per-token 🟩
`lp ≥ −1`, 🟨 `−4 ≤ lp < −1`, 🟥 `lp < −4`), then the grand total.

## Phase 4 — End-to-end pass

- Write 3–5 hand-authored puzzles with "obvious" continuations; manual check
  that their logprobs are near 0 (validates the whole pipeline).
- Full flow test: load → play set → summary.
- Clean up, confirm spec §6 run instructions work verbatim.

## Key risks / open questions

1. **Tokenizer parity** is the crux. Both sides must tokenize `text` with
   identical rules (no special adds, special-token parsing on). The parity test
   in Phase 2 should pass before UI work so we're not debugging both sides at
   once.
2. **Rust-side tokenizer choice**: llama-cpp-2's tokenization vs. an HF
   tokenizer crate — affects how trivially we match the JS tokenizer.
   Investigate first in Phase 1.2.
3. **Gemma chat template from GGUF**: need to confirm llama-cpp-2 exposes the
   embedded template; if not, fall back to Gemma's documented template format —
   worth flagging since the spec says to use the GGUF's.
4. **Logit positioning** in llama-cpp-2 (getting logits at the positions that
   predict each text token) needs a careful one-forward-pass implementation to
   avoid `expected_num_tokens − 1` separate passes.

Suggested build order rationale: parity test early (Phase 2 before 3) because
every UI behavior depends on trustworthy tokenization; inference app before UI
because the reveal state needs a live `/score` to develop against.
