# Clankerdle — Prototype Spec

> *"Everybody online is a bot, so you might as well learn to talk like one."*

Clankerdle is a daily, Wordle-style game for AI nerds and daily-puzzle fans. It gently mocks the AI world. Each challenge shows a user prompt and the beginning of an assistant reply. The player tries to type the continuation the LLM finds most likely, using an exact number of tokens.

This is a **local-only prototype** for gauging usability and fun and for iterating quickly. It will not be deployed. It has no database, no accounts and no persistence.

---

## 1. Game Flow

1. The page loads, fetches the tokenizer from the inference app and loads the bundled puzzle set.
2. The first challenge is shown. The player types a continuation.
3. The player presses Enter:
   - If the token count is wrong, the input shakes and nothing else happens.
   - If the token count is correct, the input is scored and the logprobs are revealed.
4. The player presses Enter again to go to the next challenge.
5. After the last challenge, the summary screen is shown.

Each challenge allows **one submission**. Once a guess is accepted and scored, it is final.

The game is **stateless across refreshes**. Reloading the page starts the set over at challenge 1.

---

## 2. Screens & UX

### 2.1 Visual style

The UI is very simple and plain:

- Monospace text on a plain background.
- No branding beyond a small "Clankerdle" title.
- No specific color palette is defined.

### 2.2 Challenge screen

```
User: Can you help me write a cover letter?
Assistant: Absolutely! Here's| a draft you can tailor▌

                                                 4/10
```

- **Line 1** is `User: <prompt>`.
- **Line 2** is `Assistant: <assistant_prefix><player input>`, followed by a blinking cursor.
- The context (`User:` line, `Assistant:` label and `assistant_prefix`) is **read-only**.
- The player's input appears directly after `assistant_prefix`.

**Editing**

- The player can type and delete freely within their own input.
- Ordinary text editing (arrow keys, backspace, paste) is supported within the input only.
- **Shift+Enter** inserts a newline. **Enter** submits.
- Any text is allowed, including leading or trailing whitespace and whitespace-only input. The only rule is the token count.

**Live tokenization**

- The input is re-tokenized client-side on every change. This is cheap because it involves no server call.
- Each token gets a background color from a small fixed list of pale colors. The colors cycle in order: token 1 gets color A, token 2 gets color B, and so on, looping as needed.
- Whitespace-only tokens must still be visible:
  - A space shows as a colored space.
  - A newline token shows as a `↵` glyph before the line break.
- If the player types a special-token string literally, such as `<end_of_turn>`, it is tokenized as that **single special token** and colored as one token.

**Token counter**

- A counter shows `n/N`, where `n` is the current token count and `N` is `expected_num_tokens`.

**Submitting**

- If `n ≠ N` when the player presses Enter, the input line does a short horizontal shake (about 300 ms) and the input stays editable.
- If `n = N`, the frontend sends `POST /score`. Input is locked while the request is in flight.

### 2.3 Reveal state

```
Assistant: Absolutely! Here's| a| draft| you| can| tailor| to| the| job| .|
                              ┃  ┃     ┃   ┃   ┃       ┃   ┃    ┃   ┃
                              ┃        ┃       ┃       ┃
                                       ┃       ┃
                            -0.42 -1.10 -6.31 -0.20 -3.05 ...

Score: -14.87
                                    press enter to continue...
```

**Bars**

- One bar sits under each token and is horizontally aligned with that token's span.
- The bars **hang down** from a common baseline directly under the text.
- A bar's length is proportional to `|logprob|`.
- Lengths are **clamped at −15**: any logprob ≤ −15 draws a full-length bar. The printed number is always the true value.

**Numbers**

- Each bar has its logprob printed underneath, to 2 decimal places.

**Score and continuing**

- The challenge score is the **sum of the token logprobs**, shown to 2 decimal places. Closer to 0 is better.
- `press enter to continue...` is shown. Enter advances to the next challenge, or to the summary after the last one.

The model's preferred completion and its top alternatives are **not** shown.

### 2.4 Summary screen

```
Clankerdle — results

#1   -14.87   🟩🟩🟥🟩🟨🟩🟩🟩🟩🟩
#2    -6.02   🟩🟨🟩🟩🟩
#3   -31.40   🟥🟨🟥🟩🟥🟥🟩🟨

Total: -52.29
```

- Each challenge gets one row with:
  - its index,
  - its score (sum of logprobs, 2 decimal places),
  - one emoji square per token.
- The grand total is the sum of all challenge scores.

**Emoji thresholds** (per token logprob `lp`):

| Emoji | Condition        |
|-------|------------------|
| 🟩    | `lp ≥ -1`        |
| 🟨    | `-4 ≤ lp < -1`   |
| 🟥    | `lp < -4`        |

---

## 3. Puzzles

The puzzles are hand-written in a JSON file that is **bundled into the frontend at build time**, for example `frontend/src/puzzles.json`. The server has no knowledge of puzzles.

```ts
type Puzzle = {
  id: string;                  // unique, e.g. "001"
  prompt: string;              // the user message
  assistant_prefix: string;    // beginning of the assistant reply (may be "")
  expected_num_tokens: number; // hand-chosen, >= 1
};

type PuzzleFile = Puzzle[];    // played in file order
```

For the prototype, "daily" simply means "the set in the file". There is no date-based selection.

---

## 4. Architecture

```
┌─────────────────────────┐         ┌──────────────────────────────┐
│ frontend/ (Vite,        │  HTTP   │ inference-app/ (Rust,        │
│ React + TS)             │ ──────▶ │ actix-web, llama_cpp_2)      │
│ localhost:5173          │  JSON   │ localhost:8080               │
│ - bundled puzzles.json  │         │ - Gemma 4 E2B, Q4 GGUF (CPU) │
│ - client-side tokenizer │         │ - tokenizer.json             │
└─────────────────────────┘         └──────────────────────────────┘
```

### 4.1 Frontend (`frontend/`)

- Uses the Vite `react-ts` template.
- Loads the tokenizer with `@huggingface/transformers`, using `PreTrainedTokenizer` built from the fetched `tokenizer.json` (or an equivalent library).
- While the tokenizer is loading, it shows `loading…`. If the fetch fails, it shows a plain error message.
- The inference app's base URL comes from `VITE_INFERENCE_URL` and defaults to `http://localhost:8080`.

**Tokenization rules (must match the server)**

- Tokenize **only the player's input**, independently of the context.
- Add **no** special tokens: no BOS, no EOS, and no dummy-prefix space.
- Special-token strings in the input, such as `<end_of_turn>`, **are** recognized as special tokens.
- Each token is rendered by decoding its ID individually, so the token spans concatenate back exactly to the input text.

### 4.2 Inference App (`inference-app/`)

- Rust, using `actix-web`, `actix-cors`, `serde`, `serde_json` and `llama-cpp-2`.
- Runs on the **CPU only**.
- Listens on `127.0.0.1:8080`. The port can be overridden with `PORT`.

**Model directory**

The model directory is set with `CLANKERDLE_MODEL_DIR` and defaults to `./models`. It must contain:

- exactly one `*.gguf` file (Gemma 4 E2B, Q4 quantization)
- `tokenizer.json` (the matching Hugging Face tokenizer)

**Startup**

- The model is loaded once at startup.
- If the GGUF or `tokenizer.json` is missing, startup fails with a clear error.

**Concurrency**

- A single model and context are shared behind a mutex. Requests are scored one at a time, which is fine for local use.

**CORS**

- Permissive CORS headers allow any origin, with `GET`, `POST` and `Content-Type`.

#### Prompt construction (what the model sees)

The player sees `User:` and `Assistant:`, but the model sees **Gemma's real chat template**.

1. Build the **context string**:
   - Apply the chat template embedded in the GGUF to a single user message (`prompt`), with the generation prompt enabled so the string ends at the start of the model turn.
   - Then append `assistant_prefix` verbatim.
2. Tokenize the context string with BOS added and special tokens parsed.
3. Tokenize `text` **separately**, with no BOS or EOS added and special tokens parsed. This must match the frontend rules in §4.1.
4. Concatenate the context tokens and the text tokens.
5. If the text token count is not `expected_num_tokens`, return `UnexpectedNumTokens` without running inference.
6. Run one forward pass over the full sequence, with logits requested for the positions that predict each text token.
7. For each text token `tᵢ`, compute `logprob = log_softmax(logits at the preceding position)[tᵢ]` over the **full vocabulary**. Use raw logits at temperature 1, with no sampling, top-k or top-p.

Because the context and text are tokenized separately, tokens never merge across the boundary. The frontend's count and the server's count therefore agree by construction, provided both follow the same rules. `UnexpectedNumTokens` is the server's safety net if they don't.

---

## 5. API

All requests and responses are JSON, except `GET /tokenizer`, which returns the raw `tokenizer.json`.

### 5.1 `GET /tokenizer`

- Returns the contents of `tokenizer.json` from the model directory.
- Content-Type is `application/json`.
- Status is `200`.

### 5.2 `POST /score`

**Request**

```ts
type ScoreRequest = {
  prompt: string;              // user message
  assistant_prefix: string;    // assistant reply prefix (may be "")
  text: string;                // player's continuation
  expected_num_tokens: number; // from the puzzle
};
```

**Response** (`200`)

```ts
type ScoreResponse =
  | {
      type: "Logprobs";
      logprobs: number[]; // one per text token, in order; length == expected_num_tokens
    }
  | {
      type: "UnexpectedNumTokens";
      expected_num_tokens: number;
      actual_num_tokens: number;
    };
```

The JSON is serialized with serde's internally tagged representation (`#[serde(tag = "type")]`).

**Errors**

| Status | When                                                      |
|--------|-----------------------------------------------------------|
| 400    | Malformed JSON, or `expected_num_tokens < 1`              |
| 500    | Inference failure (e.g. context too long)                 |

Errors have the body `{ "error": string }`.

The frontend treats `UnexpectedNumTokens` like a wrong count: it shakes the input and keeps it editable. On a 4xx or 5xx error it shows a plain error line and keeps the input editable.

---

## 6. Running Locally

```sh
# inference app
cd inference-app
CLANKERDLE_MODEL_DIR=../models cargo run --release

# frontend
cd frontend
npm install
npm run dev   # http://localhost:5173
```

---

## 7. Testing

- **Tokenizer parity test (important):** for a fixed list of strings, check that the frontend tokenizer and the server tokenizer produce the same token count and IDs. The list should cover:
  - leading and trailing spaces
  - newlines
  - emoji and non-ASCII text
  - `<end_of_turn>`
  - whitespace-only strings
- **Server unit tests:**
  - `UnexpectedNumTokens` when the counts differ
  - the logprobs length equals `expected_num_tokens`
  - all logprobs are ≤ 0
- **Manual check:** the logprobs for a puzzle's "obvious" continuation should be close to 0.

---

## 8. Out of Scope (for now)

- Showing the model's preferred completion or its top-k alternatives
- Multiple guesses per challenge
- Persistence of any kind (localStorage, database)
- Date-based daily puzzle rotation, and serving puzzles from the server
- A share or copy button, accounts and leaderboards
- Deployment, GPU support and mobile-specific layout
