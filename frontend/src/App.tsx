// Clankerdle — the app shell, whole-set state machine and the
// pseudo-scroll (plan 3.2 + the polish pass).
//
// App owns the tokenizer load, the current challenge index, the accepted
// submissions, and the screen *layers*:
//
//   load tokenizer → challenge 1 … challenge N → summary
//          │
//          └ loading… / plain error
//
// **The pseudo-scroll.** When the player presses Enter to continue past a
// reveal, the next screen should fade in from the bottom while the old
// one slides up and out — a crossfade with both screens briefly mounted
// at once. To keep the old one on screen, App re-renders it as a *static
// snapshot* (RevealView — a pure function of puzzle+text+logprobs, the
// same component the live challenge used) as an absolutely-positioned
// `.layer-out`, while the next challenge (or summary) mounts in `.layer`.
// After TRANSITION_MS the snapshot is unmounted. All motion is CSS
// (`fade-rise` / `fade-fall` in App.css); App only mounts/unmounts.
//
// Nothing is persisted: a refresh starts over at challenge 1 (spec section 1).

import { useEffect, useState } from "react";
import type { ReactNode } from "react";
import type { PreTrainedTokenizer } from "@huggingface/transformers";
import { GAME_NAME, INFERENCE_URL } from "./config";
import { PUZZLES } from "./puzzles";
import { loadTokenizer, tokenizeInput } from "./tokenizer";
import ChallengeView from "./ChallengeView";
import RevealView from "./RevealView";
import SummaryView from "./SummaryView";
import "./App.css";

// How long the pseudo-scroll keeps the outgoing snapshot mounted —
// slightly longer than the 350 ms CSS animations (App.css) so they
// always finish before the layer disappears.
const TRANSITION_MS = 400;

// How long the loading… layer takes to fade out once the tokenizer is
// ready (matches the `fade-out` animation on .layer-in-place-out). The
// first challenge mounts only after this — it rises *after* the loading
// text has disappeared, not beneath it (polish: loading must not scroll).
const LOADING_FADE_MS = 250;

// One accepted submission, in challenge order: the final input text (the
// snapshot re-renders its token cells) and its logprobs.
type Submission = { text: string; logprobs: number[] };

// Everything the tokenizer load can be (spec 4.1): in flight, failed, or
// ready. The game cannot start until it is ready.
type TokenizerState =
  | { kind: "loading" }
  | { kind: "error"; message: string }
  | { kind: "ready"; tokenizer: PreTrainedTokenizer };

function App() {
  const [tokenizerState, setTokenizerState] = useState<TokenizerState>({ kind: "loading" });
  // Which challenge is on screen (its index into PUZZLES), or -1 while the
  // tokenizer is still loading. The tokenizer resolves before challenge 0
  // mounts (see the effect below) so the loading layer can fade out first.
  const [challengeIndex, setChallengeIndex] = useState(-1);
  // True for the LOADING_FADE_MS during which the loading… layer fades out
  // (it drives the .layer-in-place-out class, see the loading return below).
  const [loadingFading, setLoadingFading] = useState(false);
  // One submission per challenge, appended when accepted (onAccept).
  const [submissions, setSubmissions] = useState<Submission[]>([]);
  // The challenge being animated out during a pseudo-scroll (its index),
  // or null when none. See the module comment for the mechanism.
  const [outgoing, setOutgoing] = useState<number | null>(null);

  // Fetch tokenizer.json once (spec 4.1) — the same machinery phase 2
  // proved out. React StrictMode mounts twice in dev; the effect guards
  // with `cancelled`, and a second fetch is harmless anyway.
  useEffect(() => {
    let cancelled = false;
    loadTokenizer(INFERENCE_URL)
      .then((tokenizer) => {
        if (cancelled) return;
        // The loading text fades out first; challenge 0 mounts (and rises)
        // only once it is gone — see LOADING_FADE_MS above.
        setLoadingFading(true);
        setTimeout(() => {
          if (cancelled) return;
          setLoadingFading(false);
          setTokenizerState({ kind: "ready", tokenizer });
          setChallengeIndex(0);
        }, LOADING_FADE_MS);
      })
      .catch((err: unknown) => {
        // A plain error message (spec 4.1): no retry, no stack.
        const message = err instanceof Error ? err.message : String(err);
        if (!cancelled) setTokenizerState({ kind: "error", message });
      });
    return () => {
      cancelled = true;
    };
  }, []);

  // A submission was accepted and scored (one per challenge, spec
  // section 1) — record it for the summary and the outgoing snapshot.
  const handleAccept = (text: string, logprobs: number[]) => {
    setSubmissions((previous) => [...previous, { text, logprobs }]);
  };

  // Continue past a reveal (Enter): start the pseudo-scroll. `challengeIndex`
  // here comes from this render's closure — the index of the challenge that
  // is on screen right now — so the snapshot is always *this* one.
  const handleContinue = () => {
    setOutgoing(challengeIndex);
    setChallengeIndex(challengeIndex + 1);
    setTimeout(() => setOutgoing(null), TRANSITION_MS);
  };

  if (tokenizerState.kind === "loading") {
    return (
      <Shell title={GAME_NAME}>
        {/**
          The loading layer never travels (loading text is not a puzzle):
          it fades in on mount, then — once the tokenizer arrives — fades
          back out (.layer-in-place-out), and only then does the first
          challenge rise in. No rise here (App.css has no translateY on
          these classes).
        */}
        <div className={loadingFading ? "layer-in-place-out" : "layer-in-place"}>
          <p>loading…</p>
        </div>
      </Shell>
    );
  }
  if (tokenizerState.kind === "error") {
    return (
      <Shell title={GAME_NAME}>
        <div className="layer-in-place">
          <p className="error">{tokenizerState.message}</p>
        </div>
      </Shell>
    );
  }

  const tokenizer = tokenizerState.tokenizer;
  // All challenges completed → the summary (spec 2.4).
  const inSummary = challengeIndex >= PUZZLES.length;

  return (
    <Shell title={inSummary ? `${GAME_NAME} — results` : GAME_NAME}>
      {/*
        The outgoing layer: a static snapshot of the challenge that was
        just on screen, absolutely positioned over the stage and animated
        up + out (.layer-out in App.css). It mounts fresh, so its own
        reveal animations (bars growing) are suppressed under .layer-out.
      */}
      {outgoing !== null && (
        <div className="layer layer-out" key={`out-${outgoing}`}>
          <RevealView
            puzzle={PUZZLES[outgoing]}
            tokens={tokenizeInput(tokenizer, submissions[outgoing].text)}
            logprobs={submissions[outgoing].logprobs}
          />
        </div>
      )}

      {/*
        The incoming layer: the next challenge, or the summary. The key
        remounts it per screen, so the fade-in (fade-rise) replays for
        each one.
      */}
      <div className="layer" key={`in-${challengeIndex}`}>
        {inSummary ? (
          <SummaryView results={submissions.map((s) => s.logprobs)} />
        ) : (
          <ChallengeView
            // Remounts per puzzle so the input/phase reset themselves.
            key={PUZZLES[challengeIndex].id}
            puzzle={PUZZLES[challengeIndex]}
            tokenizer={tokenizer}
            onAccept={handleAccept}
            onDone={handleContinue}
          />
        )}
      </div>
    </Shell>
  );
}

// The plain page chrome (spec 2.1): small title, monospace content.
// The .stage wraps the screens so the layers can stack (see App.css).
function Shell({ title, children }: { title: string; children: ReactNode }) {
  return (
    <main>
      <h1>{title}</h1>
      <div className="stage">{children}</div>
    </main>
  );
}

export default App;