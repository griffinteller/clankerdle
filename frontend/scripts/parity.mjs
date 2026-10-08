#!/usr/bin/env node
// Tokenizer parity test (spec section 7) — the important one.
//
// Compares, for a fixed corpus of strings, the token IDs produced by:
//   - the SERVER tokenizer: inference-app/src/bin/tokenize.rs, which calls
//     score::tokenize_text — the exact rules POST /score scores with
//   - the FRONTEND tokenizer: this script imports frontend/src/tokenizer.ts,
//     the app's real tokenization code, run on the same tokenizer.json the
//     app fetches from GET /tokenizer
//
// Parity is what makes the game playable: the counter the player sees must
// equal the count the server scores. The spec says parity holds "by
// construction" (both sides use the same rules); this test validates the
// construction.
//
// Run (from anywhere):
//     cd frontend && npm run test:parity
//
// Notes:
//   - the Rust bin loads the GGUF with vocab_only, so this takes seconds,
//     not a minute — but the first `cargo run --release` does compile.
//   - Node 23+ runs the .ts import by stripping types, so the test uses the
//     app's actual code rather than a re-implementation.

import { spawnSync } from "node:child_process";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

import { tokenizerFromJson, tokenizeInput } from "../src/tokenizer.ts";

const here = path.dirname(fileURLToPath(import.meta.url));
const frontendDir = path.resolve(here, "..");
const repoDir = path.resolve(frontendDir, "..");
const corpusPath = path.join(frontendDir, "tests", "parity-corpus.json");

// ---- 1. Server side: run the Rust tokenize bin on the corpus. ----

const cargo = spawnSync(
  "cargo",
  ["run", "--release", "--quiet", "--bin", "tokenize", "--", corpusPath],
  {
    cwd: path.join(repoDir, "inference-app"),
    env: {
      ...process.env,
      // Explicit absolute path — don't depend on the bin's cwd default.
      CLANKERDLE_MODEL_DIR: path.join(repoDir, "models"),
    },
    encoding: "utf8",
  },
);

if (cargo.status !== 0) {
  console.error("error: cargo run --bin tokenize failed:\n" + cargo.stderr);
  process.exit(1);
}

// One entry per corpus case: [{ id, text }, ...] per token.
const serverCases = JSON.parse(cargo.stdout);

// ---- 2. Frontend side: the app's tokenizer, same corpus. ----

const corpus = JSON.parse(fs.readFileSync(corpusPath, "utf8")).cases;
const tokenizerJson = JSON.parse(
  fs.readFileSync(path.join(repoDir, "models", "tokenizer.json"), "utf8"),
);
const tokenizer = tokenizerFromJson(tokenizerJson);

if (serverCases.length !== corpus.length) {
  console.error(
    `error: the bin returned ${serverCases.length} cases, the corpus has ${corpus.length}`,
  );
  process.exit(1);
}

// ---- 3. Compare. ----

let failures = 0;
corpus.forEach((text, i) => {
  const jsTokens = tokenizeInput(tokenizer, text);
  const jsIds = jsTokens.map((t) => t.id);
  const serverIds = serverCases[i].map((t) => t.id);

  const show = () => {
    const display = JSON.stringify(text);
    console.error(`case ${i} ${display}`);
    console.error(`  server (${serverIds.length}): ${serverIds.join(", ")}`);
    console.error(
      `  client (${jsIds.length}): ${jsIds.join(", ") || "(empty)"}`,
    );
    console.error(
      `  client decoded: ${JSON.stringify(jsTokens.map((t) => t.text))}`,
    );
    console.error(
      `  server decoded: ${JSON.stringify(serverCases[i].map((t) => t.text))}`,
    );
  };

  // The core check (spec 7): same token count and same IDs.
  if (jsIds.length !== serverIds.length || jsIds.some((id, k) => id !== serverIds[k])) {
    failures += 1;
    console.error("MISMATCH:");
    show();
    return;
  }

  // Sanity check for the UI (spec 4.1): decoding each ID individually and
  // concatenating must reproduce the input exactly.
  const roundTrip = jsTokens.map((t) => t.text).join("");
  if (roundTrip !== text) {
    failures += 1;
    console.error("ROUND-TRIP BROKEN:");
    console.error(`  input:     ${JSON.stringify(text)}`);
    console.error(`  rejoined:  ${JSON.stringify(roundTrip)}`);
    return;
  }
});

if (failures > 0) {
  console.error(`\nparity FAILED: ${failures}/${corpus.length} cases differ.`);
  process.exit(1);
}

console.log(`parity OK: ${corpus.length} cases, frontend and server agree.`);
console.log(`  corpus: ${path.relative(frontendDir, corpusPath)}`);