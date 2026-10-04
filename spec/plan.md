# Clankerdle — Prototype Implementation Plan

Based on `spec/prd.md`, with stack and hosting preferences from `spec/notes.md`.

**Key decisions (the PRD has been updated to match):**
- Puzzles and all config (global rules, share config, model registry, tokenizers) live in **Postgres**, not in repo YAML files. YAML is still the authoring format, but the CLI imports it into the database. The database is the source of truth. Publishing a puzzle is a database operation, not a deploy.
- The launch model is **Gemma 4 E2B, base (pretrained) checkpoint**, not the `-it` instruction-tuned one, because the game is about base models.
- **For now, the worker runs Gemma 4 E2B unquantized (bf16) on a GPU**, on Runpod Serverless GPU workers, using **HF transformers**. The weights are baked into the image.
  - **Why GPU:** cold start is roughly the same as on CPU, and warm requests and beam search are much faster.
  - **Why unquantized:** it's the simplest option (no quantization step or extra dependency) and gives the most faithful scores. It costs a few seconds of cold start.
  - **Cold starts:** the endpoint **scales to zero** (no always-on worker). The client **wakes the worker as soon as a player opens a playable puzzle**, so the cold start mostly overlaps with reading and typing (§5.6).
  - **Why transformers over llama.cpp:** it's simpler here (§5.1). It works directly with Google's base checkpoint and its tokenizer, and the beam loop is plain PyTorch.
  - **Why not vLLM:** still out. Our load is one forward pass per guess plus one cached beam search per completion, so its throughput features don't help.
  - **Changing later:** quantization, engine, and hardware are per-model settings behind an engine interface in the worker. Another quantization, llama.cpp, or another GPU type needs no Rust or game changes (§5.7).
- **No mock backend and no CI.** Local dev calls the real Runpod endpoint. Automated tests use a test-only fake that the server can't select. Everything, including the worker image, is built and tested locally by hand.
- **Free Render plans for the prototype.** The API may take up to a minute to wake, and the database is backed up by hand with `pg_dump` (§11). Upgrade before any wider deploy.

## 1. Stack summary

| Layer | Choice | Hosting |
|---|---|---|
| Web client | Vite + React + TypeScript (SPA, no server components), Vitest + Testing Library | Render static site |
| Game API + admin CLI | Rust, actix-web, tokio, sqlx (Postgres), `tokenizers` crate, reqwest, serde (+ YAML for import). Tests use Rust's built-in test harness. | Render web service (Docker, free plan for now) |
| Storage | Postgres: puzzles, config, models, tokenizers, beam cache, submission log | Render Postgres (free plan for now) |
| Inference | Custom Python worker: PyTorch + HF transformers, serving Gemma 4 E2B base unquantized (bf16), for now (see §5) | Runpod Serverless (GPU workers, scale to zero) |

## 2. Repository layout

```
clankerdle/
  server/                  # Rust crate: lib + 2 bins
    Dockerfile             # API image for Render (multi-stage; see §11)
    .sqlx/                 # sqlx offline query data, so Docker builds don't need a live DB
    src/
      lib.rs
      config.rs            # AppConfig types, rule merging
      puzzle.rs            # Puzzle / Completion / ModelConfig types, YAML authoring schema
      repo/                # PuzzleRepository + ConfigRepository traits; Postgres + in-memory impls
      inference/           # InferenceBackend trait + RunpodBackend (the only implementation)
      tokenize.rs          # TokenizerRegistry, template rendering, round-trip check
      modes/               # Scorer, Objective, Mode traits + registry
      store/               # BeamCache + SubmissionLog traits; Postgres + in-memory impls
      api/                 # actix handlers, DTOs, errors
      validate.rs          # puzzle validation (used by the CLI on push/publish and by the repo on load)
      bin/server.rs        # HTTP server (runs migrations on boot)
      bin/clankerdle.rs    # admin CLI (see §9)
    migrations/            # sqlx migrations: schema + the default global config row (§4)
    tests/                 # integration tests (API, DB, beam pipeline, RunpodBackend job handling)
      support/fake_backend.rs  # test-only FakeBackend with canned logprobs and beams; the server can't select it
      fixtures/
        puzzles/*.yaml     # example puzzles for tests and local dev
        tokenizer.json     # small, permissively licensed tokenizer (e.g. GPT-2), tests only
  worker/                  # Runpod serverless handler (Python)
    handler.py             # Runpod entrypoint: op dispatch, input validation, weights check
    sampling.py            # process_logits (PyTorch, fp32), shared by every engine; tested against a NumPy reference
    ops.py                 # score and the beam loop, engine-agnostic
    engines/
      base.py              # Engine interface (§5.2)
      transformers.py      # PyTorch + transformers, bf16
    Dockerfile             # CUDA worker image for Runpod Serverless, bf16 weights baked in as a cached layer (§5.3)
    build.sh               # local build and push (there's no CI)
    requirements.txt, tests/
  web/                     # Vite react-ts app
  fixtures/
    tokenizer-parity/      # golden strings -> Gemma token IDs, shared by Rust and Vitest
  docker-compose.local-postgres.yml  # local dev and test Postgres only (not used in prod)
  render.yaml              # Render blueprint: API, static site, Postgres
  scripts/test.sh          # runs every test suite locally (there's no CI)
```

## 3. Domain model (Rust)

```rust
struct ModelConfig { name: String,
                     hf_repo: String, revision: String,          // base checkpoint; source of the tokenizer
                     tokenizer_sha256: String,
                     engine: Engine,                             // Transformers now; LlamaCpp possible later
                     weights_uri: String, weights_sha256: String, // e.g. the pinned Google checkpoint
                     quantization: String,                       // e.g. "bf16"
                     runpod_endpoint_id: String, template: String, add_bos: bool, sampling: Sampling }
struct Sampling   { temperature: f32, top_p: Option<f32>, top_k: Option<u32> }
struct Rules      { guesses_per_completion: u32, attempt_scoring: AttemptScoring /* best|last */,
                    reveal_top_k: u32, beam_width: u32, reveal_timing: RevealTiming /* per_completion|end */,
                    show_model: ShowModel /* before|after|never */ }
struct Completion { index: usize, prompt: String, partial: String, tokens: usize,
                    target_logprob: Option<f64> }
struct Puzzle     { id: String, number: u32, mode: String, content_hash: String,
                    model: ModelConfig /* registry defaults ← puzzle overrides */,
                    rule_overrides: PartialRules, completions: Vec<Completion> }
struct AppConfig  { id: i64 /* config version */, rules: Rules, scoring: ScoringConfig, share: ShareConfig }
```

Effective rules are computed **at read time** (`config.rules ← puzzle.rule_overrides`). Changing global config affects every puzzle that doesn't override that field, without editing any puzzle.

The traits follow the PRD. `async_trait` is used because they're object-safe and held as `Arc<dyn …>`:

```rust
trait PuzzleRepository {               // read side, used by the game API
    async fn current(&self) -> Result<Arc<Puzzle>>;            // highest-numbered *published* puzzle
    async fn get(&self, id: &str) -> Result<Arc<Puzzle>>;
}
trait PuzzleAdmin {                    // write side, used only by the CLI
    async fn upsert_draft(&self, p: &PuzzleInput) -> Result<Puzzle>;
    async fn publish(&self, id: &str) -> Result<()>;
    async fn unpublish(&self, id: &str) -> Result<()>;
    async fn list(&self) -> Result<Vec<PuzzleSummary>>;
}
trait ConfigRepository {
    async fn current(&self) -> Result<Arc<AppConfig>>;         // latest config row
    async fn model(&self, name: &str) -> Result<Arc<ModelConfig>>;
    async fn tokenizer_json(&self, sha256: &str) -> Result<Arc<Vec<u8>>>;
}

trait InferenceBackend {
    async fn score_tokens(&self, model: &ModelConfig, context: &[u32], guess: &[u32]) -> Result<Vec<f64>>;
    async fn beam_search(&self, model: &ModelConfig, context: &[u32], n: usize, width: u32) -> Result<Vec<Beam>>;
    async fn warm(&self, model: &ModelConfig) -> Result<()>;   // fire-and-forget wake-up (§5.6); debounced by the caller
}

trait Scorer    { async fn score(&self, ctx: &ScoringCtx, c: &Completion, guess: &[u32]) -> Result<ScoredGuess>; }
trait Objective { async fn reference(&self, ctx: &ScoringCtx, c: &Completion) -> Result<f64>;
                  fn score(&self, value: f64, reference: f64) -> f64;
                  fn beat_reference(&self, score: f64) -> bool { false } }
struct Mode     { id: &'static str, scorer: Arc<dyn Scorer>, objective: Arc<dyn Objective> }
```

- Tokenization moved out of `InferenceBackend` into a `TokenizerRegistry`. It loads `tokenizer.json` bytes from `ConfigRepository` by sha256, builds a `tokenizers::Tokenizer`, and caches it forever (the content is addressed by its hash, so it never changes). The PRD's `tokenize` method still exists, but as a facade over the registry.
- `RunpodBackend` is the only `InferenceBackend`, and local dev uses it too (§11). Tests use a `FakeBackend` that lives in `server/tests/support/` and can't be selected by the server.
- **Caching:** the Postgres repositories keep an in-memory cache with a short TTL (`CACHE_TTL_SECS`, default 15). Publishing a puzzle or changing config takes effect within seconds, with no redeploy and without a DB round-trip on every request. LISTEN/NOTIFY could replace this later, but it isn't needed now.
- `ScoredGuess { token_ids, token_logprobs: Vec<f64>, value: f64 /* L */, truncated: Vec<bool> }`
- `ModeRegistry` maps a mode id string to a `Mode`. Mode ids live in code, not the DB, because a mode is logic. Validation rejects unknown ids.
  - `highest_probability` = `LogprobScorer` + `BeamGapObjective` (reference = cached L_beam, score = ref − value, beat = score < −ε)
  - `target_probability` = `LogprobScorer` + `TargetObjective` (reference = `target_logprob`, score = |value − ref|)
  - A future reward-model mode adds a new `Scorer` and reuses these objectives without changes elsewhere.
- Scores are in nats (natural log). ε (`scoring.beat_epsilon`, default 1e-3) absorbs float noise.

### Tokenization rules (`tokenize.rs`)
- `render(template, prompt, partial)` uses literal `{prompt}` / `{partial}` replacement, not a format engine, so braces in content are safe.
- Context IDs = `encode(rendered, add_special = add_bos)`. Gemma expects `<bos>`, so `add_bos: true`. A test asserts the first context ID is the BOS id.
- Guess IDs = `encode(guess_text, add_special = false)`, appended to the context IDs. The guess is never tokenized together with the partial.
- `round_trips(ids)`: `encode(decode(ids), false) == ids`. Used to filter beams. This matters more for Gemma: its byte-fallback tokens and large vocab produce many token sequences that decode to the same text.
- Any token the sampling settings truncate away (top-p/top-k) has logprob −∞. It's clamped to `scoring.logprob_floor` (default −50) and marked `truncated`.

## 4. Data model (Postgres)

All tables are created by sqlx migrations: numbered SQL files in `server/migrations/`, applied in order and tracked in sqlx's bookkeeping table. Schema changes always add a new file; existing ones are never edited. The API runs `sqlx::migrate!` on boot, so no Render pre-deploy command is needed.

Migrations also insert the **default global config** as the first `app_config` row (contents in §4.1), so a fresh database always has a working config. Models and puzzles are never created by migrations. They depend on the environment (Runpod endpoint, tokenizer download) and are added with the CLI (§9). Until they exist, the site shows the not-initialized screen (§7, §10).

```sql
-- Tokenizers are content-addressed blobs (stored gzip-compressed).
create table tokenizers (
  sha256 text primary key, source text not null,          -- e.g. "hf:google/gemma-4-e2b@<rev>"
  json_gz bytea not null, created_at timestamptz default now());

-- Model registry. "Known model" = an enabled row here.
create table models (
  name text primary key,                                   -- e.g. "gemma-4-e2b-bf16"
  hf_repo text not null, revision text not null,           -- base checkpoint (tokenizer source)
  tokenizer_sha256 text not null references tokenizers,
  engine text not null default 'transformers',             -- transformers now; llama_cpp possible later
  weights_uri text not null,                               -- e.g. "hf:google/gemma-4-e2b@<rev>"
  weights_sha256 text not null,                            -- the worker rejects requests for other weights
  quantization text not null,                              -- e.g. "bf16" now; "bnb-nf4" etc. later
  runpod_endpoint_id text not null,                        -- the GPU endpoint serving exactly these weights
  default_template text not null, add_bos bool not null default true,
  default_sampling jsonb not null,                         -- {"temperature":1.0,"top_p":null,"top_k":null}
  enabled bool not null default true, created_at timestamptz default now());

-- Global config, append-only. The latest row is active; history is kept for playtest analysis.
create table app_config (
  id bigserial primary key,
  rules jsonb not null, scoring jsonb not null, share jsonb not null,
  note text, created_by text, created_at timestamptz default now());

create table puzzles (
  id text primary key, number int not null unique check (number > 0),
  mode text not null,
  model_name text not null references models,
  sampling jsonb, template text,                           -- null = use model defaults
  rule_overrides jsonb not null default '{}',
  status text not null default 'draft' check (status in ('draft','published')),
  published_at timestamptz,
  content_hash text not null,                              -- hash of everything that affects scoring
  created_at timestamptz default now(), updated_at timestamptz default now());
create index on puzzles (number desc) where status = 'published';

create table completions (
  puzzle_id text references puzzles on delete cascade, idx int not null,
  prompt text not null, partial text not null,
  tokens int not null check (tokens > 0),
  target_logprob float8,
  primary key (puzzle_id, idx));

create table beam_cache (
  key text primary key, puzzle_id text, completion_index int,
  beams jsonb not null,
  worker_version text not null,                            -- the worker that computed them; not part of the key (§6) created_at timestamptz default now());

create table submissions (
  id bigserial primary key, player_id uuid not null,
  puzzle_id text not null references puzzles, puzzle_hash text not null, config_id bigint not null,
  completion_index int not null, attempt int not null,
  guess_text text not null, token_ids int[] not null, token_logprobs float8[] not null,
  value float8 not null, reference float8 not null, score float8 not null,
  beat_reference bool not null, mode text not null, model text not null, worker_version text not null,
  created_at timestamptz default now());
create index on submissions (puzzle_id, player_id, completion_index);
```

- **Current puzzle:** `where status = 'published' order by number desc limit 1`.
- Submissions record `puzzle_hash`, `config_id`, and `worker_version`. If a dev edits a live puzzle, changes the bands, or moves a model to new hardware mid-playtest, the logs still show what each guess was scored against.
- Editing a published puzzle is allowed, but the CLI warns when `content_hash` changes and submissions already exist.
- The JSON shape of `rules`, `scoring`, and `share` mirrors §4.1. The Rust types use `serde(deny_unknown_fields)`, so bad config is rejected when it's written, not when it's read.
- A `#[sqlx::test]` asserts that the default config row inserted by the migration deserializes into `AppConfig`, so the SQL and the Rust types can't drift apart. Adding a config field means writing a migration that appends a new `app_config` row containing it, or giving the field a serde default.

### 4.1 Default global config

This is inserted by the migration, shown here as YAML. Change it later with `clankerdle config get > config.yaml`, edit the file, then `clankerdle config set -f config.yaml`:

```yaml
rules:
  guesses_per_completion: 1
  attempt_scoring: best        # best | last  (only matters when guesses > 1)
  reveal_top_k: 5
  beam_width: 16
  reveal_timing: per_completion # per_completion | end
  show_model: after            # before | after | never
scoring:
  beat_epsilon: 0.001
  logprob_floor: -50
  max_tokens: 16               # validation upper bound for N
share:
  formatter: default
  title: "Clankerdle #{number}"
  bands:                       # per mode; score <= max -> emoji
    highest_probability: [{max: 1.0, emoji: "🟩"}, {max: 4.0, emoji: "🟨"}, {max: .inf, emoji: "🟥"}]
    target_probability:  [{max: 0.5, emoji: "🟩"}, {max: 2.0, emoji: "🟨"}, {max: .inf, emoji: "🟥"}]
  clanker_scale: {highest_probability: 3.0, target_probability: 1.5}  # pct = mean(exp(-max(score,0)/scale))
  summary: "{pct}% clanker"
```

### 4.2 Registering a model

Models are added with CLI flags. There's no model file in the repo. The launch model, unquantized on GPU:

```sh
clankerdle model add \
  --name gemma-4-e2b-bf16 \
  --hf-repo google/gemma-4-e2b \           # TODO confirm the exact HF id of the *base* (non -it) checkpoint
  --revision <pinned commit sha> \         # pin it; an upstream change would silently change every score
  --engine transformers --quantization bf16 \
  --weights hf:google/gemma-4-e2b@<pinned rev> \         # same checkpoint; baked into the worker image (§5.3)
  --runpod-endpoint <id> \                 # the GPU endpoint serving exactly these weights
  --template $'User: {prompt}\n\nAssistant: {partial}' \
  --add-bos true --temperature 1.0         # top-p/top-k default to none
```

- By default `model add` downloads `tokenizer.json` at the pinned revision with `HF_TOKEN`, because Gemma is gated behind a license acceptance. It stores the file in `tokenizers` and then upserts the `models` row.
- `weights_sha256` is a hash over the checkpoint's file hashes, read from Hugging Face's file metadata, so the multi-GB files aren't downloaded. `--weights-dir <path>` hashes a local directory instead.
- **Each quantization or engine is its own model row**: `gemma-4-e2b-bf16` now, and maybe `gemma-4-e2b-nf4` (4-bit) later. Scores from different rows aren't comparable. Moving a puzzle to another row is a re-push and re-publish, which recomputes its beams.
- **Hardware isn't part of the row's identity.** Moving the same weights to an endpoint with a different GPU type is a `model add` upsert with the new `--runpod-endpoint`, followed by `clankerdle beams refresh` (§5.7).
- `--tokenizer-file <path>` loads a local file instead of downloading one.
- `--runpod-endpoint` is required. Every model is served by a Runpod endpoint, including in local dev.
- `PLAYTEST.md` records the exact command used for each prod model.

## 5. Inference worker (Runpod Serverless)

The worker is a custom Python Runpod handler. **For now it runs Gemma 4 E2B base, unquantized (bf16), on a GPU, using PyTorch and HF transformers.** There's no serving framework, because we need exact control over logprob processing and fixed-length beams. The endpoint **scales to zero** (no always-on worker). Cold starts are hidden by waking the worker when a player opens a playable puzzle (§5.6).

Each endpoint serves exactly one model row, and its weights are fixed when the image is built. The worker accepts three operations:

```jsonc
// score
{"op": "score", "model": "gemma-4-e2b-bf16", "weights_sha256": "...", "sampling": {...},
 "context": [ids], "guess": [ids]}            -> {"logprobs": [f64...]}
// beam
{"op": "beam", "model": "...", "weights_sha256": "...", "sampling": {...},
 "context": [ids], "n": 4, "width": 16}       -> {"beams": [{"ids": [...], "logprobs": [...], "sum": f64}, ...]}  // all `width` finals
// ping (wake-on-load, §5.6)
{"op": "ping"}                                -> {"ok": true}
// every response also carries
// {"worker_version": "<image tag>+torch<ver>+transformers<ver>+cuda<ver>+<GPU name>"}
```

If a `score` or `beam` request's `weights_sha256` doesn't match the loaded weights, the worker returns an error. That way a misconfigured endpoint can't silently score against the wrong weights.

### 5.1 Why transformers and not llama.cpp

Transformers is simpler here:

- **Weights exist as-is.** We use Google's base checkpoint directly, with no conversion or quantization step. With llama.cpp we'd need a GGUF of the *base* model, and published GGUFs are often `-it` only.
- **Architecture support.** New Gemma releases are usually supported in transformers on day one. llama.cpp support for E-series models can lag.
- **No vocab mapping.** The model and `tokenizer.json` come from the same HF repo, so token IDs match by construction.
- **Simpler beam search.** Beams are just the batch dimension, and the KV cache is reordered with `index_select`. llama.cpp would need its low-level multi-sequence KV API.
- **Easy testing.** The same code path runs on a tiny HF model on CPU.

The cost is a bigger image (CUDA PyTorch plus about 10 GB of bf16 weights) and slower cold starts, roughly 15–30 s by our current estimate. Wake-on-load is meant to hide that. If it doesn't hide enough, the fallbacks are 4-bit weights or llama.cpp behind the same engine interface (§5.7).

### 5.2 Engine interface

Sampling and beam logic sit outside the engine. Changing quantization or hardware only touches the engine or the weights.

```python
class Engine(Protocol):
    special_ids: set[int]                                              # never chosen in beams
    def prefill(self, context: list[int]) -> tuple[State, Tensor]     # KV state after context, next-token logits
    def extend(self, state: State, tokens: Tensor) -> Tensor           # [beams, len] tokens -> fp32 logits after each
    def select(self, state: State, beam_idx: Tensor) -> State          # reorder/duplicate beams (KV index_select)
```

`TransformersEngine` is the only engine for now.

### 5.3 Weights

- **Source:** Google's Gemma 4 E2B base checkpoint at the pinned revision, in bf16, loaded with `AutoModelForCausalLM`, `torch_dtype=bfloat16`, and `device_map="cuda"`, straight from safetensors.
- **Text only:** if the checkpoint is multimodal, a CPU-only step in the Docker build loads the text decoder and saves it with `save_pretrained`, dropping the vision and audio towers. That keeps the image smaller and the load faster. No GPU is needed at build time.
- **Baked in, as a cached layer:** `worker/Dockerfile` has a separate `weights` build stage. It downloads the pinned checkpoint (`HF_TOKEN` build secret, since Gemma is gated) and does the text-only extraction with CPU PyTorch, pinned to the worker's versions. The final CUDA PyTorch runtime image copies the weights in right after the base image, before installing Python packages and copying the worker code.
  - On a code or dependency change, Docker reuses the cached weights layer. It's rebuilt only when the pinned revision, the extraction script, or the base image changes.
  - `docker push` skips layers the registry already has, so a rebuild uploads only the small code layers. Runpod hosts that have the previous image pull only those.
  - The `weights` stage runs on the build machine's native platform (`--platform=$BUILDPLATFORM`), since its output is just files. That keeps the extraction fast when building the linux/amd64 image on an Apple Silicon Mac.
  - Weights stay on the worker's local disk, so cold starts don't read them over the network. A Runpod network volume would avoid even the first upload, but it ties the endpoint to one datacenter and loads weights more slowly, so we don't use one.
- **`weights_sha256`:** a hash over the source checkpoint's file hashes, taken from Hugging Face's metadata. The text-only extraction is deterministic under the pinned library versions. The build verifies the downloaded files against it and records it in a manifest in the image. The worker reads the manifest at startup, without re-hashing GBs of files, and compares it with each request's `weights_sha256`.

### 5.4 Operations

- **Sampling semantics:** one function, `process_logits(logits, sampling) -> logprobs`, is shared by both scoring ops:
  1. Cast the logits to fp32.
  2. Divide by the temperature.
  3. Apply the top-k mask.
  4. Apply the top-p mask, computed on the softmax of what's left.
  5. Take `log_softmax`. Masked tokens come out as −∞.
- **Score:** a single forward pass over `context + guess` with batch size 1. Take the logits at positions `len(context)-1 … len(context)+n-2`, run them through `process_logits`, and gather the guess token at each position. A forward pass over a few hundred tokens takes tens of milliseconds on a GPU.
- **Beam search:** a custom loop of about 100 lines. We don't use `generate()`, because its beam scorer, length penalty, and EOS handling don't fit our rules.
  - Prefill the context once. Then for exactly `n` steps:
    - Score each live beam's next-token logits with `process_logits`.
    - Take the top `width` candidates over all beams × vocab by cumulative logprob, never choosing −∞ candidates or `special_ids` (EOS and friends, which players can't type).
    - `select` the parent beams and `extend` them by one token, as one batch.
  - Return all `width` final beams with per-token logprobs.
  - The worker doesn't filter round-trips. Rust does that with the authoritative tokenizer, then re-scores the survivors through `score` (§6). That re-scoring is what makes L_beam exactly comparable to player scores, since batched beam steps can differ from batch-size-1 scoring in the last bits.
- **Ping:** returns immediately. The model is loaded at process start, before the handler accepts jobs, so a worker that answers a ping is ready to score.
- **Token IDs:** the worker never tokenizes text. It takes IDs from Rust, and the context already includes `<bos>`, so the worker must not add one.

### 5.5 Runtime

- **Reproducibility:**
  - `model.eval()`, `torch.inference_mode()`, fp32 log-softmax, TF32 off, and deterministic cuBLAS settings.
  - Always score with batch size 1, so a player's guess and the re-scored beam go through identical kernels.
  - Pin the `torch`, `transformers`, and CUDA versions in the image.
  - The endpoint is restricted to **one GPU type**. Runpod can fall back to other GPU types, but different GPUs can shift scores slightly, and the drift could be on the order of `beat_epsilon`.
  - A test asserts that scoring the same input twice is bit-identical.
- **Hardware:** a single common **24 GB** GPU type with good serverless availability (e.g. RTX A5000, L4, or RTX 4090/3090). The ~10 GB of bf16 weights would fit on 16 GB, but 24 GB leaves headroom for width-16 beams. Phase 0 picks the type.
- **Latency and job handling:** `RunpodBackend` calls `/runsync` for scores and beams.
  - If Runpod returns before the job finishes (status `IN_QUEUE` or `IN_PROGRESS`, e.g. during a long cold start), it polls `/status/{id}` until the job completes or a deadline passes (`INFERENCE_DEADLINE_SECS`, default 120; the CLI uses a longer one).
  - It retries once, and only if the request fails before Runpod accepts the job, so it never queues a duplicate.
  - Beams are fast on a GPU (about a second per completion). The UI shows a themed "waking the clanker…" state for slow submissions.
- **Tests:** pytest with a tiny HF model on CPU (e.g. `sshleifer/tiny-gpt2`). It's the same code path as production, just a different checkpoint. The tests check:
  - `process_logits` against a NumPy reference for temperature, top-k, and top-p.
  - Score logprobs against a naive token-by-token forward loop.
  - Beam search: exactly `n` tokens per beam, results sorted, `width` beams returned, no special tokens.
  - Beam search with width 1 matches greedy decoding.
  - Each beam's sum matches `score` on its ids, within 1e-4.
  - Determinism.
  - A weights mismatch is rejected.
  - `ping` returns only after the model has loaded.

  GPU smoke tests run against the real Gemma on the Runpod endpoint and are opt-in (`scripts/test.sh --gpu`).
- **Local dev:** there's no local worker mode. Local dev calls the Runpod endpoint just like prod (§11). To try a new worker image before prod, deploy it to a second dev endpoint and point the model row in your local database at it.

### 5.6 Cold starts: scale to zero + wake-on-load

There's no always-on worker. A cold start (container start, PyTorch import, loading ~10 GB onto the GPU, first-pass warm-up) is estimated at about 15–30 s. Players usually spend 10–30 s reading and typing before their first submit, so the worker is woken as soon as they open a playable puzzle.

- **Endpoint settings:** min/active workers **0**, idle timeout **~10 minutes** (so a session stays warm between guesses), FlashBoot on, a small max worker count (e.g. 3).
- **Client:** after loading the current puzzle, if the player hasn't finished it (i.e. it isn't a results screen), the client calls `POST /api/warm` (§7) once, fire-and-forget. It calls again when the tab becomes visible or the input is focused after more than `idle timeout` of inactivity.
- **API:**
  - `/api/warm` calls `InferenceBackend::warm(model)`, which submits a `ping` through Runpod's async `/run`. That starts a worker without the API holding a connection open.
  - It's debounced per model: no ping if one was sent, or a real request finished, within the last 60 s (`WARM_DEBOUNCE_SECS`). Reloads and many simultaneous players therefore cost at most one ping a minute.
  - It's also capped per day (`WARM_MAX_PER_DAY`, default 100). Past the cap, warm does nothing and guesses just pay the cold start, so hitting `/api/warm` in a loop can't keep a GPU awake around the clock. Both counters live in memory.
- **Publishing:** the CLI pings before `publish` and `beams refresh` and waits for the worker. Only devs ever see that cold start.
- **Cost:** roughly one worker start plus a ~10-minute idle tail per playtest session, with nothing billed between sessions.
- **Decision rule:** if Phase 0 shows that players still wait noticeably on their first guess even with wake-on-load, move to 4-bit weights or llama.cpp (§5.7). An always-on worker is the last resort.

### 5.7 Later: other quantizations, engines, or hardware

Nothing outside the worker changes for any of these.

- **Quantized weights** (e.g. 4-bit NF4 via bitsandbytes, roughly 3 GB, likely a faster cold start):
  - Add a one-off quantize script and host the artifact in a private HF repo.
  - Build an image with it and deploy it as a new endpoint.
  - Register it as a **new model row**, e.g. `model add --name gemma-4-e2b-nf4 --weights … --quantization bnb-nf4`.
  - Puzzles pick the row by name. Scores from different rows aren't comparable, which is why a quantization is never swapped in place.
- **llama.cpp engine** (lighter import and faster load, if cold start is still a problem): add a `LlamaCppEngine` behind the same interface with a GGUF artifact. It's a new model row.
- **Different GPU type:** point the model row at an endpoint with the new GPU type, then run `clankerdle beams refresh`. Cached beams don't invalidate themselves (§6), so the refresh is required. Scores shift slightly, so don't switch during a playtest.

## 6. Beam cache and precompute

- **Cache key:** sha256 of `(model name, weights_sha256, sampling, context token IDs, n, width)`. Editing a prompt, template, or sampling setting therefore invalidates the cache automatically.
- **Worker version:** the `worker_version` that computed the beams (library versions plus GPU name) is stored next to them, not in the key, so the API can look beams up without asking the worker first.
  - When a guess's score response reports a different `worker_version` than the cached beams (after a worker rebuild or a GPU change), the API logs a warning and keeps using the cached beams. It never recomputes on its own, so L_beam can't shift in the middle of a playtest.
  - `clankerdle status` lists puzzles with stale beams, and `clankerdle beams refresh` recomputes them.
- **Cache value:** filtered, re-scored beams, stored as JSONB in `beam_cache`.
- **Pipeline:** `beam_search` → keep beams where `round_trips(ids)` holds → **re-score each survivor with `score_tokens`** so L_beam comes from exactly the same code path as player scores → sort by sum descending → store.
- If zero beams survive, the completion is invalid and `publish` fails. If fewer than `reveal_top_k` survive, show what's there and print a warning.
- **When it runs:** as part of `clankerdle puzzle publish` (see §9). Precompute is tied to publishing rather than deploys, so a puzzle can't go live without its beams.
  - **Fallback:** if the cache is missing (e.g. `beam_width` changed in global config after publish), the first request computes it behind a per-key single-flight lock (`tokio::sync::Mutex` map plus `INSERT … ON CONFLICT DO NOTHING`). That makes the first player wait (worst case through a cold start), so `config set` warns when a change affects beams and offers to run `beams refresh`.
  - `clankerdle beams refresh` recomputes for all published puzzles.
- The reference value L_beam = `beams[0].sum`. Target mode doesn't use beams for scoring, but still precomputes them for "reveal top answers".

## 7. Game API

All routes live under `/api`. CORS allows only the static site origin (from env). The API has **no write endpoints for puzzles or config**. All authoring goes through the CLI with direct DB access (§9).

| Method & path | Purpose |
|---|---|
| `GET /api/puzzles/current` | The latest published puzzle. Returns **404** `{error: "no_puzzle"}` if there's no valid published puzzle (fresh database, no model registered yet, or everything unpublished), and **503** `{error: "unavailable"}` if the database is unreachable. The client shows an error screen for each (§10). Otherwise the response contains public fields only: id, number, mode, completions (prompt, partial, tokens, `target_logprob` in target mode only), effective rules, share config, `tokenizer_id` (sha256), and the model name if `show_model == before`. |
| `GET /api/tokenizers/{sha256}` | Serves the stored gzip blob as-is with `Content-Encoding: gzip` and `Cache-Control: public, max-age=31536000, immutable`. The ID is opaque, so the model name isn't leaked when it's hidden. |
| `POST /api/puzzles/{id}/completions/{idx}/guesses` | Body `{player_id, text}`. The server re-tokenizes and returns **422** `{error: "wrong_token_count", got, expected}` if the count ≠ N, and **409** if the player has no guesses left. Otherwise it scores, logs (with `puzzle_hash`, `config_id`, and `worker_version`), and returns `{tokens: [{id, text, logprob, truncated}], value, reference, score, beat_reference, attempt, attempts_left, top_beams?}`. `top_beams` comes back only on the final attempt, and the client hides it behind "reveal top answers". The model name is included if `show_model == after`. Guessing on a draft puzzle returns 404. If inference fails or misses its deadline (§5.5), it returns **503** `{error: "inference_unavailable"}`. Nothing is logged and the attempt isn't used up, so the player can retry. |
| `POST /api/warm` | Wake-on-load (§5.6). The client calls it when it shows a playable puzzle. The API fires `InferenceBackend::warm` for the current puzzle's model in the background, debounced per model (`WARM_DEBOUNCE_SECS`, default 60) and capped per day (`WARM_MAX_PER_DAY`, default 100), and returns **202** immediately. It has no body, needs no player ID, and never fails visibly to the player. |
| `GET /api/healthz` | Liveness check plus a DB ping. Also reports `{initialized, missing: ["model" \| "published_puzzle"]}` for devs. An uninitialized database doesn't make it fail, so Render doesn't restart-loop a fresh deploy. |

- Token display text comes from encoding offsets into the guess string, not per-token decode. Gemma's byte-fallback tokens would otherwise render as garbage for emoji and other multi-byte characters.
- Errors are a single `ApiError` enum mapped to JSON.
- When `PgPuzzleRepository` loads a puzzle, it runs the cheap offline validation. If the latest published puzzle is somehow invalid, it logs an error and serves the next-highest valid one rather than breaking the site. If none is valid, the endpoint returns `no_puzzle`.
- The server always boots on an empty or partly set-up database. At startup it logs a warning listing what's missing.

## 8. Authoring and validation

Devs still hand-write puzzles in the PRD's YAML format. The only differences are that `model.name` refers to a `models` row, and `model.sampling` and `model.template` are optional overrides. The YAML files can live anywhere (a scratch dir, a gist, `server/tests/fixtures/puzzles/` for examples). They aren't read by the server.

Validation (`validate.rs`) runs on `push`, again on `publish`, and in a cheap form on load:
- The schema parses (unknown fields are warnings, since fields a mode doesn't use are allowed).
- `id` and `number` are unique. The DB constraints enforce this too.
- The mode is in the registry. The model is an enabled `models` row.
- There are at least 2 completions. Each has `1 ≤ tokens ≤ scoring.max_tokens`. Target mode requires `target_logprob < 0` on every completion.
- The effective template contains `{prompt}` and `{partial}`.
- **Achievable token count** (on publish, with inference): at least one round-tripping beam of length N exists.

## 9. Admin CLI (`clankerdle`)

The CLI connects with `DATABASE_URL`: the local dev Postgres or Render's external connection string. Commands that need inference also need `RUNPOD_API_KEY`, and `model add` needs `HF_TOKEN` to download Gemma's tokenizer.

```
clankerdle migrate                                # also runs on server boot; handy before the first CLI use
clankerdle status                                 # what's missing (model, published puzzle; same as /api/healthz) + puzzles with stale beams (§6)
clankerdle config get                             # print active config as YAML
clankerdle config set -f config.yaml [--note ".."]  # validate + append new app_config row
clankerdle model add --name .. [flags, §4.2]      # fetch (or --tokenizer-file) tokenizer, upsert model
clankerdle model list
clankerdle puzzle push -f 0012.yaml [--model <name>]  # validate (offline) + upsert as draft; warns on hash change
                                                  # --model overrides model.name (e.g. fixtures against a local model)
clankerdle puzzle publish <id> [--yes]            # validate + precompute beams + set status=published
clankerdle puzzle unpublish <id>
clankerdle puzzle list
clankerdle puzzle export <id> > 0012.yaml         # DB -> YAML round-trip for editing
clankerdle puzzle preview <id>                    # rendered context, token counts, top beams, L_beam
clankerdle score <id> <idx> "<text>"              # debug-score a guess (no logging)
clankerdle beams refresh [--puzzle <id>]
```

Publishing a puzzle is `push`, then `preview`, then `publish`. It's live for players within `CACHE_TTL_SECS`.

## 10. Web client

**State and identity**
- `player_id` is generated with `crypto.randomUUID()` on first visit and stored in localStorage.
- localStorage also holds `progress[puzzleId]`, used to resume mid-puzzle, and `finished[puzzleId] = PuzzleResult`.
- On load, if `finished[current.id]` exists, show the results screen. Otherwise resume or start.
- Game flow is a pure reducer (`gameReducer`) that reads the effective rules. It's unit-tested in isolation.

**Tokenizer (Gemma-specific concerns)**
- `Tokenizer` is a TS interface `{ encode(text): {ids, offsets} }`.
- **Size:** Gemma's ~262k-entry vocab makes `tokenizer.json` large (likely tens of MB uncompressed, ~10 MB gzipped; measure it in Phase 0). **We accept that cost** and always tokenize locally. Since the file is immutable, players download it once per model.
  - It's served gzipped and cached as immutable, so later visits load it from the browser's HTTP cache.
  - The download starts as soon as the puzzle JSON arrives, but only for a playable puzzle. Players who already finished it go straight to the results screen, which doesn't need the tokenizer.
  - Until the tokenizer is ready, a themed full-page **"getting the puzzle ready…"** state shows instead of the puzzle. On cached visits, that's only the parse time. The same state covers the `GET /api/puzzles/current` request itself, which can take up to a minute while the free-plan API wakes (§11).
  - Wake-on-load (§5.6) fires when the puzzle JSON arrives, without waiting for the tokenizer, so the tokenizer download overlaps with the GPU cold start.
  - If the tokenizer fails to load, the client shows `ErrorScreen`'s "the clanker is down" state with a retry button.
  - Parsing and encoding run in a **Web Worker**, so the main thread never blocks. `encode` is async via `postMessage`, and stale results are dropped by sequence number.
  - If gzip is still too big, a later option is a compact pre-processed format. That's not in scope now.
- **Implementation:** default to `@huggingface/transformers`' tokenizer loaded from `tokenizer.json`. Phase 0 spike: compare it against the Rust `tokenizers` crate compiled to WASM (`unstable_wasm`) on the parity fixtures, plus load and encode time with the Gemma file. Pick WASM if the JS library fails parity (Gemma's normalizer and byte-fallback are the likely trouble spots) or is too slow.
- **Parity fixtures:** `fixtures/tokenizer-parity/gemma-4-e2b.json` holds only strings and expected IDs, generated once with Python `tokenizers` at the pinned revision. The tokenizer file itself isn't committed (gated license). `scripts/test.sh` fetches it with `HF_TOKEN`. The fixtures cover tricky strings: leading, double and trailing spaces, newlines, tabs, emoji, CJK, digits, contractions, and `▁` literally.

**Token-guided input (`TokenInput`)**
- A transparent `<textarea>` sits over a mirrored highlight layer. It re-encodes on each input event via the worker.
- Segments come from offsets and get alternating tints. Spaces are drawn as a faint `·` and newlines as `↵`. Gemma is expected not to add a dummy prefix (a Phase 0 parity check), so `" the"` and `"the"` are different tokens, and the dot makes that visible.
- Shows a counter `3 / 5 tokens` that goes into an error state when the count ≠ N. Submit is disabled unless the count == N and the latest encode result matches the current text. Enter submits.
- A server 422 is shown inline. That case would indicate a parity bug and gets logged to the console.

**Screens and components**
- `Header`: number, mode, target (target mode), model name per `show_model`.
- `CompletionCard`: prompt, partial, `TokenInput`, submit, and the reveal (or "locked until end" if `reveal_timing == end`).
- `Reveal`: per-token chips colored and scaled by `exp(logprob)` on a perceptual scale, with exact values on hover or tap and truncated tokens flagged. The score bar is drawn against the reference. There's a beat-the-beam flourish, and `TopAnswers` is a disclosure.
- `Results`: total, per-completion scores, `ShareButton` (Clipboard API with a fallback).
- `ErrorScreen`: themed full-page states. For `no_puzzle`: "no clanker here yet", with a hint for devs to run `clankerdle status`. For `unavailable` or a network failure: "the clanker is down", with a retry button.
- `ShareFormatter` is a TS interface. A registry of formatters is selected by `share.formatter`. `DefaultShareFormatter(shareConfig)` builds the title, band emoji, and summary line. Because share config comes from the DB, copy and bands change without any deploy.

**API client**
- A typed `fetch` wrapper with `VITE_API_URL`. Slow submissions (>2 s) show "waking the clanker…".
- A 503 `inference_unavailable` on a guess keeps the text in the input and offers a retry. The attempt isn't spent.
- **Wake-on-load:** `POST /api/warm` once the current puzzle loads and isn't already finished, again when the tab becomes visible, and on input focus after more than ~10 minutes idle (§5.6). It's fire-and-forget and errors are ignored.

## 11. Deployment (Render + Runpod)

`render.yaml` defines:
- **`clankerdle-api`**: Docker web service built from `server/Dockerfile`, with `server/` as the build context. Migrations are embedded in the binary at compile time, so nothing outside `server/` is needed. The Dockerfile has two stages:
  - A `rust` builder stage compiles the release binaries `server` and `clankerdle`, with `SQLX_OFFLINE=true` so the build uses the committed `.sqlx/` data instead of a live database. It caches dependencies (e.g. with `cargo-chef`).
  - A slim Debian runtime stage holds both binaries and CA certificates. The CLI is in the image so it can be run from a Render shell.

  The image contains no puzzles, models, or tokenizers. Everything comes from the DB.
  - Env: `DATABASE_URL`, `RUNPOD_API_KEY`, `WEB_ORIGIN`, `CACHE_TTL_SECS`, `WARM_DEBOUNCE_SECS`, `WARM_MAX_PER_DAY`, `INFERENCE_DEADLINE_SECS`. The CLI reads `DATABASE_URL`, `RUNPOD_API_KEY`, and `HF_TOKEN` from each dev's `.env`.
  - Migrations run on boot, so pre-deploy isn't needed.
  - **Free plan for the prototype.** The API sleeps when idle and takes up to about a minute to wake. Wake-on-load only fires once the API is up, so for the first visitor the two waits add up. The "getting the puzzle ready…" state covers it. Upgrade before any wider deploy.
- **`clankerdle-web`**: static site, `npm ci && npm run build`, publish `web/dist`, SPA rewrite `/* → /index.html`, env `VITE_API_URL`.
- **`clankerdle-db`**: Render Postgres, free plan for the prototype. Free databases have no backups and expire about 30 days after creation (check Render's current terms). It's the only copy of the puzzles, config, and submissions, so:
  - A dev runs `pg_dump` against the external connection string at least weekly and after each publishing session, and keeps the dumps outside Render.
  - Before the expiry date, create a new free database, restore the latest dump, and point the API at it. Or upgrade.
  - `clankerdle puzzle export` stays as a YAML escape hatch for individual puzzles.
- **DB access for devs:** Render's external connection string with an IP allowlist, kept in each dev's local `.env` (never committed).
- **Runpod worker image:** the CUDA image from `worker/Dockerfile`, with the pinned bf16 text-only weights baked in (an image of roughly 16–18 GB). It's built locally and pushed by hand with `worker/build.sh` (there's no CI), on a machine with roughly 40 GB of free disk. Thanks to the cached weights layer (§5.3), only the first push uploads the weights. It's deployed as a Runpod Serverless **GPU** endpoint restricted to one GPU type, using the scale-to-zero settings in §5.6. Its endpoint ID goes in the `models` row.

**Bootstrapping prod:**
1. Deploy the services. Migrations run on boot and insert the default config, and the site shows the "no clanker here yet" screen.
2. `clankerdle model add --name gemma-4-e2b-bf16 …` (§4.2).
3. Push and publish the first puzzles. `clankerdle status` should report nothing missing.

**Local dev setup:**
1. `docker compose -f docker-compose.local-postgres.yml up -d`. This starts only Postgres, pinned to the same major version as Render's, with a named volume and port 5432.
2. Run the server with `RUNPOD_API_KEY` set. Migrations and the default config apply on boot.
3. Register the model with the §4.2 command (needs `HF_TOKEN`). Point `--runpod-endpoint` at the prod endpoint, or at a dev endpoint when testing a new worker image. There's no mock or local worker, so local dev always runs real Gemma on Runpod and pays its cold starts.
4. `clankerdle puzzle push -f server/tests/fixtures/puzzles/0001.yaml --model gemma-4-e2b-bf16`, then `clankerdle puzzle publish 0001`, which computes beams on Runpod.
5. Run `npm run dev` in `web/`.

`#[sqlx::test]` uses the local dev Postgres instance. sqlx creates and drops a throwaway database per test, so test runs never touch the dev data.

## 12. Testing strategy

| Area | Tests |
|---|---|
| Rust unit (`#[cfg(test)]`) | Rule merging; YAML → `PuzzleInput` parsing and validation errors; `content_hash` stability; template rendering; objectives (gap, abs distance, ε); round-trip filter; −∞ clamping |
| Rust DB (`#[sqlx::test]`, ephemeral DB per test) | Migrations apply; the default config row deserializes into `AppConfig`; `status` reports missing model/puzzle on a fresh DB; `PgPuzzleRepository::current` ignores drafts and picks the highest number; config versioning (latest row wins); tokenizer blob round-trip; publish fails when no beam survives (with `FakeBackend`); `status` lists stale beams; submission logging including `puzzle_hash`/`config_id`/`worker_version`; guess counting |
| Rust integration (`server/tests/`) | Beam pipeline with `FakeBackend` (filter, re-score, sort) and the worker-version mismatch warning. `RunpodBackend` against a test-only stub HTTP server: `/runsync` returning `IN_PROGRESS` then polling `/status`, the deadline, and no retry once a job is accepted. `actix_web::test` with in-memory repos, `FakeBackend`, and the fixture tokenizer and puzzles: response shape (target hidden in highest mode, model hidden per config), 422/409/404 paths, `no_puzzle` on an empty repo, `unavailable` when the repo errors, `top_beams` only on final attempt, `/api/warm` returns 202, is debounced (one `warm` call for many requests), and stops at the daily cap; an inference failure returns 503 `inference_unavailable` without logging or using up the attempt |
| Gemma tokenizer | Parity fixtures in Rust and Vitest (needs `HF_TOKEN`); BOS-first assertion |
| Worker | pytest with a tiny HF model on CPU (§5.5); opt-in GPU smoke tests against the Runpod endpoint |
| Web (Vitest) | `gameReducer`; `DefaultShareFormatter`; segmentation from offsets; `TokenInput` gating, including stale-encode protection; the "getting the puzzle ready…" state until the tokenizer resolves, and the error state if it fails; no tokenizer download or loading state for a finished puzzle; a 503 on a guess offers a retry and keeps the attempt; results restore from localStorage; warm is called for a playable puzzle but not for a finished one; `ErrorScreen` for `no_puzzle`, `unavailable`, and network errors |
| Manual E2E | Local dev setup (§11) against the Runpod endpoint, then the deployed site |

There's no CI. `scripts/test.sh` runs everything locally against the compose Postgres: `cargo fmt --check`, `clippy -D warnings`, `cargo test`, offline validation of `server/tests/fixtures/puzzles/*.yaml`, `npm run lint`, `tsc --noEmit`, `vitest run`, and the worker pytest on CPU. `--gpu` adds the worker smoke tests against Runpod. Devs run it before merging.

## 13. Milestones

Each phase ends in something demoable. Rough sizes assume one dev.

**Phase 0 — Spikes and scaffolding (3 days)**
- Repo layout, `scripts/test.sh`, `docker-compose.local-postgres.yml`, `server/Dockerfile`, `render.yaml` stub.
- Confirm the exact HF id of the Gemma 4 E2B base checkpoint, accept its license, and pin a revision.
- **Model loading:** confirm that transformers loads the checkpoint as a text-only causal LM in bf16, and record the minimum `transformers` version. Prototype the Dockerfile's `weights` stage (CPU-only text-only extraction) and confirm that a code-only rebuild reuses the weights layer and pushes only the small layers.
- **Runpod GPU smoke test:** a throwaway handler (with `ping`) on a scale-to-zero endpoint that loads the model and returns logprobs for one forward pass.
  - Measure image size.
  - Measure true cold start: scaled to zero, FlashBoot miss, timed to first response.
  - Measure cold start on a FlashBoot hit.
  - Measure warm latency for a score call and for an n=8, width=16 beam.
  - Confirm that Runpod downloads the image when the endpoint is deployed and not on each cold start.
  - Simulate wake-on-load (ping, then score 15 s later) and record how long the score still waits. That number drives the §5.6 decision rule.
  - Pick the single 24 GB GPU type based on serverless availability.
- Tokenizer spike: Gemma `tokenizer.json` size (raw and gzip), whether it adds a dummy prefix, and transformers.js vs. `tokenizers` WASM for parity plus load and encode time in a Web Worker.
- *Exit:* decisions are recorded in `spec/decisions.md`, and one Gemma score call works end to end from a script.

**Phase 1 — Rust core and DB (4–5 days)**
- Migrations (schema and default config); types; config merge; Postgres and in-memory repos with TTL cache; `TokenizerRegistry`; mode registry with both modes; test-only `FakeBackend`; validation; fixture tokenizer and puzzles; CLI `migrate`/`status`/`config`/`model add|list`/`puzzle push|list|export`; unit and `sqlx::test` tests.
- *Exit:* on a fresh local DB, `clankerdle status` reports what's missing. After `model add` and `puzzle push`, `clankerdle puzzle preview` shows the rendered context and token counts (no inference yet), and the `FakeBackend` tests pass.

**Phase 2 — Inference and beams (2–3 days)**
- Worker (`Engine` interface and `TransformersEngine`, `process_logits`, `score`, custom beam loop, `ping`, weights check), pytest tests, and its CUDA image and `build.sh`; `RunpodBackend` including `/status` polling, deadlines, and `warm`; CLI pings before publish; `clankerdle score`; stale-beam reporting in `status`; beam pipeline and `BeamCache`; `puzzle publish` with precompute; `puzzle preview`; `beams refresh`.
- *Exit:* two sample puzzles are published with real Gemma L_beam values computed on the Runpod GPU endpoint, and the CLI scores real guesses.

**Phase 3 — Game API (2 days)**
- Routes (including debounced and capped `/api/warm`), DTOs, errors (including 503 `inference_unavailable`), `SubmissionLog`, guess limits, CORS, gzip tokenizer serving, integration tests.
- *Exit:* the full puzzle can be played via `curl`, and publishing a new puzzle with the CLI switches `current` within the TTL.

**Phase 4 — Web client core loop (3–4 days)**
- API client with wake-on-load, player ID and progress storage, Web Worker tokenizer, `TokenInput`, `CompletionCard`, `gameReducer`, a basic reveal, results screen.
- *Exit:* the full puzzle is playable in the browser against the Runpod endpoint, and refresh resumes correctly.

**Phase 5 — Reveal and share polish (2–3 days)**
- Probability-shaded token visualization, the beat-the-beam flourish, `TopAnswers`, `DefaultShareFormatter`, Wordle-winking copy, an acceptable mobile layout.
- *Exit:* the reveal feels good, and changing the bands in DB config changes the share text without a deploy.

**Phase 6 — Deploy and playtest readiness (1–2 days)**
- Render blueprint live on free plans, the first `pg_dump` backup taken, prod bootstrapped (§11), Runpod endpoint wired in, 3–5 puzzles published across both modes.
- A `PLAYTEST.md` with the publish workflow and SQL snippets: beam-beat rate per completion, score distributions by `config_id`, per-token logprob outliers, guesses per player.
- *Exit:* devs play the latest puzzle on the deployed URL. Publishing is `push` + `publish`, with no PR or deploy.

**Total: about 4 weeks.**

## 14. How the open questions map to config

| PRD open question | Knob (DB-backed; no API or web deploy needed) |
|---|---|
| Guesses per completion and the feedback between them | `app_config.rules.guesses_per_completion` / `attempt_scoring`, or per-puzzle `rule_overrides` |
| Which mode is more fun | `puzzles.mode` |
| Target granularity | Per-completion now. A puzzle-level target would be a new `Objective` plus a puzzle-level `target_logprob`. Not built yet. |
| Token count N | `completions.tokens` |
| Model, quantization, and temperature; show model before/after | `models` rows (one per quantization/engine; a new one also needs a worker image and endpoint), `puzzles.sampling`, `rules.show_model` |
| Reveal timing | `rules.reveal_timing` |
| Score bands and % clanker | `app_config.share` |
| Beam-beat rate and beam width | `rules.beam_width`; `submissions.beat_reference` |
| Release cadence | `clankerdle puzzle publish`, whenever |
| Chat template | `models.default_template`, `puzzles.template` |

## 15. Risks and assumptions

- **Gemma 4 E2B specifics are unverified.** The exact HF id, base-checkpoint availability, the minimum transformers version, tokenizer size, whether the tokenizer adds a dummy prefix, and multimodal packaging (and how cleanly the text decoder can be extracted) all need checking. All of these are Phase 0 checks. The fallback for the client tokenizer is WASM.
- **Cold starts with scale-to-zero.**
  - The estimate is 15–30 s for a true cold start with bf16 weights. Wake-on-load should hide most of it, but fast players may still wait a few seconds on their first guess.
  - The image is roughly 16–18 GB, so a cold start that also has to download it (e.g. a worker moved to a new machine) could take a minute or more.
  - If the chosen GPU type is scarce, requests can queue for a free worker.
  - On Render's free plan, the API's own wake-up (up to about a minute) comes first and adds to it.
  - Mitigations: wake-on-load, a ~10-minute idle timeout, FlashBoot, a lean runtime image, text-only weights, and a widely available GPU type. Fallbacks, in order: 4-bit weights, llama.cpp, and an always-on worker as a last resort (§5.6–5.7).
- **Wake-on-load can over-wake.** Every visit to a playable puzzle may start a GPU worker, but the debounce limits it to one ping a minute and `WARM_MAX_PER_DAY` caps the total. Visitors who never submit still cost an idle-timeout tail, which is negligible at playtest scale.
- **Small model, flat distributions.** A 2B-effective base model may be less "opinionated" than an 8B one, which affects puzzle difficulty and beam-beat rate. This is a playtest question, and swapping models is a `models` row plus a worker image.
- **Large client tokenizer.** The first load (~10 MB gzipped) is slow on mobile. This is an accepted cost: a "getting the puzzle ready…" state covers it once per model, and later visits use the cache. Also mitigated by gzip, the Web Worker, and overlapping the download with wake-on-load. The fallback is tokenizing through the API, which we considered and rejected for now because of input delay.
- **The DB is the single source of truth.** On the free plan there are no Render backups, and the database expires after about 30 days. Mitigated by regular `pg_dump`s kept off Render, a restore before expiry, and `puzzle export`. Direct prod DB access for devs is acceptable for an internal prototype only.
- **Throughput.** Each worker serves one request at a time, so concurrent guesses queue or spin up more Runpod workers. Requests are fast on a GPU, so that's fine for internal playtests. Revisit (batching, or a vLLM backend behind the same interface) only if a public launch happens.
- **Numerical drift across hardware.** Scores can differ slightly between GPU types or library versions. Mitigations:
  - The endpoint is restricted to one GPU type, and the torch, transformers, and CUDA versions are pinned.
  - Beams are re-scored through the same path as player guesses.
  - `worker_version` is stored with cached beams and logged with every submission. A mismatch logs a warning and shows up in `clankerdle status`.
  - After a rebuild or hardware change, run `clankerdle beams refresh`.
- **No CI.** Tests run only when a dev runs `scripts/test.sh`, so regressions can slip in. That's acceptable for a small team. Add CI before any wider launch.
- **Local dev depends on Runpod.** There's no offline development, and devs pay cold starts and a little GPU time while developing. The Rust, web, and worker tests still run without Runpod.
- **Beam post-filtering** can discard every top beam. `beam_width` is the knob for that.
- **Gemma license:** serving `tokenizer.json` to browsers counts as redistributing part of a Gemma model. That should be fine under the Gemma terms for an internal prototype, but check before any public launch.
- **Assumption:** the `{prompt}` / `{partial}` template is the only chat-rendering mechanism, and the partial is never trimmed.
