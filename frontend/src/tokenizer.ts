// Client-side tokenizer (spec section 4.1).
//
// Loads `tokenizer.json` from the inference app and tokenizes the player's
// input on every keystroke — no server call, so it is cheap.
//
// The rules here MUST match the server exactly (spec 4.1):
//   - tokenize ONLY the player's input, never the context
//   - add NO special tokens: no BOS, no EOS, no dummy-prefix space
//   - special-token strings typed literally (e.g. `<turn|>`) become a
//     single special token
//
// The parity test (scripts/parity.mjs) pins these rules against the server,
// and the server's `UnexpectedNumTokens` response is the runtime safety net.
//
// This module is plain TS with no Vite-specific imports, so the parity test
// can import it from Node directly (Node 23+ runs .ts files by stripping
// types) — the test exercises the app's real tokenization code, not a copy.

import { PreTrainedTokenizer } from "@huggingface/transformers";

// One token of the player's input.
//
// `text` is the token's ID decoded individually, so joining the `text`s of
// all tokens reproduces the input exactly (spec 4.1: "each token is
// rendered by decoding its ID individually"). The UI (phase 3) renders one
// colored span per token using `text`.
export type Token = {
  id: number;
  text: string;
};

// Build the tokenizer from parsed tokenizer.json contents.
//
// Exported separately from `loadTokenizer` so the parity test can build the
// tokenizer from the file on disk (byte-identical to GET /tokenizer) without
// needing the server running.
//
// The second constructor argument is tokenizer_config.json contents, which we
// don't have; an empty object works — the special tokens we care about come
// from the tokenizer.json `added_tokens` list.
export function tokenizerFromJson(json: unknown): PreTrainedTokenizer {
  return new PreTrainedTokenizer(json as object, {});
}

// Fetch tokenizer.json from the inference app (spec 5.1) and build the
// tokenizer. Throws on any failure; the caller (App) shows the error.
export async function loadTokenizer(baseUrl: string): Promise<PreTrainedTokenizer> {
  const response = await fetch(`${baseUrl}/tokenizer`);
  if (!response.ok) {
    throw new Error(`GET ${baseUrl}/tokenizer failed: HTTP ${response.status}`);
  }
  return tokenizerFromJson(await response.json());
}

// Tokenize the player's input with the spec 4.1 rules: one Token per input
// token, in order.
export function tokenizeInput(
  tokenizer: PreTrainedTokenizer,
  text: string,
): Token[] {
  // add_special_tokens: false -> the post-processor (which adds BOS etc.)
  // is skipped; special-token strings in the text are still recognized,
  // because they live in the tokenizer's added_tokens list.
  const ids: number[] = tokenizer.encode(text, { add_special_tokens: false });

  // Decode each ID on its own so the spans concatenate back to exactly the
  // input text. skip_special_tokens: false keeps special tokens (and their
  // spellings) visible.
  return ids.map((id) => ({
    id,
    text: tokenizer.decode([id], { skip_special_tokens: false }),
  }));
}