#!/usr/bin/env node
// Smoke-render test (companion to the parity test).
//
// The parity test checks tokenization; this checks the *screens*: it
// server-renders the main components with the real tokenizer + puzzle
// data (via Vite's SSR transform, so .tsx and CSS imports just work)
// and asserts the resulting HTML is shaped like the spec's screens.
// Effects (focus, key listeners) don't run server-side — this is a
// render check, not an interaction test; those stay manual.
//
// Run (from anywhere):
//     cd frontend && npm run test:render

import { readFile } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { createElement } from "react";
import { renderToString } from "react-dom/server";
import { createServer } from "vite";

const here = path.dirname(fileURLToPath(import.meta.url));
const frontendDir = path.resolve(here, "..");
const repoDir = path.resolve(frontendDir, "..");

// Vite in middleware mode transforms the .tsx modules on demand.
const vite = await createServer({
  root: frontendDir,
  logLevel: "error",
  server: { middlewareMode: true },
});
try {
  const puzzlesMod = await vite.ssrLoadModule("/src/puzzles.ts");
  const tokenizerMod = await vite.ssrLoadModule("/src/tokenizer.ts");
  const App = (await vite.ssrLoadModule("/src/App.tsx")).default;
  const ChallengeView = (await vite.ssrLoadModule("/src/ChallengeView.tsx")).default;
  const RevealView = (await vite.ssrLoadModule("/src/RevealView.tsx")).default;
  const SummaryView = (await vite.ssrLoadModule("/src/SummaryView.tsx")).default;

  const tokenizer = tokenizerMod.tokenizerFromJson(
    JSON.parse(await readFile(path.join(repoDir, "models", "tokenizer.json"), "utf8")),
  );
  const puzzles = puzzlesMod.PUZZLES;
  const noop = () => {};

  // React SSR separates adjacent text nodes with `<!-- -->`; strip them so
  // checks can look for plain strings like "0/10".
  const norm = (html) => html.replaceAll("<!-- -->", "");

  // ---- 1. App, initial state: the loading layer. ----
  const appHtml = norm(renderToString(createElement(App)));
  check(appHtml.includes("loading…"), "App renders the loading… layer");
  check(appHtml.includes("layer"), "App wraps screens in a .layer");

  // The heading shows the game name from the one config variable
  // (config.ts: GAME_NAME) — spec 2.1's only branding.
  const configMod = await vite.ssrLoadModule("/src/config.ts");
  check(appHtml.includes(configMod.GAME_NAME), "App titles screens with GAME_NAME");

  // ---- 2. ChallengeView, playing: context lines, colored cells, counter. ----
  const challengeHtml = norm(
    renderToString(
      createElement(ChallengeView, {
        puzzle: puzzles[0],
        tokenizer,
        onAccept: noop,
        onDone: noop,
      }),
    ),
  );
  check(challengeHtml.includes("User: "), "challenge shows the User: line");
  check(challengeHtml.includes("Assistant: "), "challenge shows the Assistant: line");
  check(
    challengeHtml.includes("0/" + puzzles[0].expected_num_tokens),
    "challenge shows the n/N counter",
  );
  check(
    challengeHtml.includes('class="cursor"'),
    "challenge shows the blinking cursor",
  );
  check(
    challengeHtml.includes('class="hidden-input"'),
    "challenge mounts the invisible textarea",
  );

  // With input, tokens render as cycled colored cells. ChallengeView's
  // input is internal state (not drivable without a browser), so render
  // the line component directly with tokens.
  const { AssistantLine } = await vite.ssrLoadModule("/src/AssistantLine.tsx");
  const tokens = tokenizerMod.tokenizeInput(tokenizer, " Hello world");
  const lineHtml = norm(
    renderToString(
      createElement(AssistantLine, {
        puzzle: puzzles[0],
        lines: (await vite.ssrLoadModule("/src/display.ts")).displayLines(tokens),
        showCursor: true,
      }),
    ),
  );
  check(
    lineHtml.includes("token-0") && lineHtml.includes("token-1"),
    "input tokens render as cycled colored cells",
  );
  check(lineHtml.includes(" Hello"), "token text is rendered from the token spans");

  // ---- 3. RevealView: bars, true numbers, score, hint. ----
  // (logprobs length always equals the token count — spec 5.2 — so the
  // test data keeps them matched.)
  const revealHtml = norm(
    renderToString(
      createElement(RevealView, {
        puzzle: puzzles[0],
        tokens,
        logprobs: [-0.42, -1.1],
      }),
    ),
  );
  check(revealHtml.includes('class="bar"'), "reveal renders bars");
  check(revealHtml.includes("-0.42") && revealHtml.includes("-1.10"),
    "reveal prints true logprobs");
  check(revealHtml.includes("Score: -1.52"), "reveal sums the logprobs into the score");
  check(revealHtml.includes("press enter to continue..."), "reveal shows the hint");

  // Clamping (spec 2.3): a logprob ≤ -15 draws a full-length bar, but the
  // printed number stays the true value.
  const clampedHtml = norm(
    renderToString(
      createElement(RevealView, {
        puzzle: puzzles[0],
        tokens,
        logprobs: [-20.5, -0.42],
      }),
    ),
  );
  check(clampedHtml.includes("height:64px"), "logprob ≤ -15 renders a full-length bar");
  check(clampedHtml.includes("-20.50"), "the printed number stays the true value");

  // ---- 4. SummaryView: one row per challenge, emoji, total. ----
  const summaryHtml = norm(
    renderToString(
      createElement(SummaryView, {
        results: [
          [-0.5, -2, -5], // 🟩🟨🟥
          [-10, -0.1], // 🟥🟩
        ],
      }),
    ),
  );
  check(summaryHtml.includes("#1") && summaryHtml.includes("#2"), "summary numbers rows");
  check(summaryHtml.includes("🟩") && summaryHtml.includes("🟨") && summaryHtml.includes("🟥"),
    "summary picks emoji by the spec 2.4 thresholds");
  check(summaryHtml.includes("Total:"), "summary shows the grand total");

  console.log("render smoke OK: challenge, reveal, summary and loading screens render.");
} finally {
  await vite.close();
}

function check(ok, what) {
  if (!ok) {
    console.error(`render smoke FAILED: ${what}`);
    process.exit(1);
  }
  console.log(`  ok: ${what}`);
}