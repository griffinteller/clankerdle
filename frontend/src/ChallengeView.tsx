// The challenge screen (plan 3.3–3.6): the two context lines, the live
// tokenized input, the n/N counter — and, once scored, the reveal.
//
// This component owns the per-challenge lifecycle (plan 3.2):
//
//   playing   typing; Enter submits
//   scoring   input locked, POST /score in flight
//   revealed  RevealView (bars + score); Enter continues via onDone
//
// App remounts ChallengeView per challenge (React `key`), so the input and
// phase reset themselves; one submission per challenge is enforced simply by
// moving to `revealed` and never coming back (spec section 1). The
// pseudo-scroll between screens is App's job (see App.tsx).
//
// Input editing (spec 2.2): a real <textarea> does all the native work
// (typing, arrows, backspace, paste, Shift+Enter) but is kept invisible;
// the player "types into" the rendered token spans. This is far simpler
// and more robust than re-implementing text editing by hand.

import { useEffect, useMemo, useRef, useState } from "react";
import type { KeyboardEvent as ReactKeyboardEvent } from "react";
import type { PreTrainedTokenizer } from "@huggingface/transformers";
import { postScore, type ScoreRequest } from "./api";
import { INFERENCE_URL } from "./config";
import { AssistantLine, UserLine } from "./AssistantLine";
import { displayLines } from "./display";
import type { Puzzle } from "./puzzles";
import RevealView from "./RevealView";
import { tokenizeInput } from "./tokenizer";

// The shake duration (spec 2.2: "about 300 ms"). Only the animation timing
// lives here; the keyframes live in App.css as `@keyframes shake`.
const SHAKE_MS = 300;

// Per-challenge lifecycle (see the module comment).
type Phase = "playing" | "scoring" | "revealed";

type Props = {
  puzzle: Puzzle;
  tokenizer: PreTrainedTokenizer;
  // Called once, when the submission is accepted and scored: App records
  // it (it needs the final text to re-render this screen as the outgoing
  // snapshot of the pseudo-scroll, plus the logprobs for the summary).
  onAccept: (text: string, logprobs: number[]) => void;
  // Called once, when the player presses Enter on the reveal screen: App
  // starts the pseudo-scroll to the next challenge (or the summary).
  onDone: () => void;
};

export default function ChallengeView({ puzzle, tokenizer, onAccept, onDone }: Props) {
  const [input, setInput] = useState("");
  const [phase, setPhase] = useState<Phase>("playing");
  // The accepted submission's logprobs — set together with phase
  // "revealed", one per text token.
  const [logprobs, setLogprobs] = useState<number[] | null>(null);
  // A plain error line from a failed POST (spec 5.2); null when hidden.
  const [error, setError] = useState<string | null>(null);

  // The invisible textarea the player actually types into.
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  // The assistant line — the element that shakes (spec 2.2).
  const lineRef = useRef<HTMLDivElement>(null);
  // Guards against events racing ahead of re-renders (see `submit` and the
  // reveal listener below): a second Enter before React re-renders must
  // not fire a second request / advance twice.
  const lockRef = useRef(false);
  const doneRef = useRef(false);

  // Re-tokenize the whole input on every change (spec 2.2). Cheap: no
  // server involved. StrictMode double-renders are harmless (pure call).
  const tokens = useMemo(() => tokenizeInput(tokenizer, input), [tokenizer, input]);
  const lines = useMemo(() => displayLines(tokens), [tokens]);

  // ---- Interactions ----

  // ~300 ms horizontal shake (spec 2.2). Restarting a CSS animation from
  // React means toggling the class off and on with a reflow in between;
  // doing that imperatively on a ref is simpler than threading a
  // "shake counter" through state. The class is only cosmetic, so React
  // never needs to know it's there.
  const shake = () => {
    const el = lineRef.current;
    if (!el) return;
    el.classList.remove("shake");
    void el.offsetWidth; // force a reflow so the animation restarts
    el.classList.add("shake");
    setTimeout(() => el.classList.remove("shake"), SHAKE_MS);
  };

  // Enter submits (spec 2.2). Called from the textarea's keydown, so all
  // other keys are native editing — no interference needed.
  const submit = async () => {
    // Ignore Enter while a request is in flight or after the reveal:
    // `phase` may be stale for one event, `lockRef`/`doneRef` never are.
    if (lockRef.current || doneRef.current) return;

    const n = tokens.length;
    if (n !== puzzle.expected_num_tokens) {
      // Wrong token count: shake, keep editing (spec 2.2).
      shake();
      return;
    }

    // Lock the input while the request is in flight (spec 2.2).
    lockRef.current = true;
    setPhase("scoring");
    setError(null);

    // The request payload (spec 5.2): the puzzle's context + the input.
    const request: ScoreRequest = {
      prompt: puzzle.prompt,
      assistant_prefix: puzzle.assistant_prefix,
      text: input,
      expected_num_tokens: puzzle.expected_num_tokens,
    };

    try {
      const response = await postScore(INFERENCE_URL, request);
      if (response.type === "UnexpectedNumTokens") {
        // The server counted differently than we did — the parity safety
        // net (spec 4.2). Treat it exactly like a wrong count (spec 2.2).
        lockRef.current = false;
        setPhase("playing");
        shake();
        return;
      }
      // Accepted and scored: this submission is final (spec section 1).
      setLogprobs(response.logprobs);
      setPhase("revealed");
      // Hand the submission to App (see onAccept above).
      onAccept(input, response.logprobs);
    } catch (e) {
      // 4xx/5xx: plain error line, input stays editable (spec 5.2).
      lockRef.current = false;
      setPhase("playing");
      setError(e instanceof Error ? e.message : String(e));
    }
  };

  const onKeyDown = (event: ReactKeyboardEvent<HTMLTextAreaElement>) => {
    if (event.key !== "Enter") return;
    if (event.shiftKey) return; // Shift+Enter: native newline (spec 2.2)
    event.preventDefault(); // plain Enter is ours: submit (spec 2.2)
    void submit();
  };

  // Focus follows the phase: into the textarea whenever the player can
  // type (mount and after an error), out of it once revealed so Enter
  // reaches the window listener below instead of the textarea.
  useEffect(() => {
    if (phase === "playing") textareaRef.current?.focus({ preventScroll: true });
    if (phase === "revealed") textareaRef.current?.blur();
  }, [phase]);

  // `press enter to continue...` (spec 2.3): a window listener while
  // revealed. `doneRef` makes continuing a one-shot even if a second
  // Enter lands before App re-renders this component away.
  useEffect(() => {
    if (phase !== "revealed") return;
    const onKey = (event: KeyboardEvent) => {
      if (event.key !== "Enter" || doneRef.current) return;
      if (!logprobs) return; // cannot happen in "revealed", but stay honest
      event.preventDefault();
      doneRef.current = true;
      onDone();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [phase, logprobs, onDone]);

  // ---- Rendering ----

  // Clicking anywhere on the challenge refocuses the (invisible) input
  // while the player can type (spec 2.2) — so the whole area behaves like
  // one big text field. The ref is only touched inside the handler, never
  // during render.
  const refocus = () => {
    if (phase === "playing") textareaRef.current?.focus({ preventScroll: true });
  };

  return (
    <div className="challenge" onClick={refocus}>
      {phase === "revealed" && logprobs !== null ? (
        // The reveal (spec 2.3) — the same component App re-renders as the
        // outgoing snapshot of the pseudo-scroll.
        <RevealView puzzle={puzzle} tokens={tokens} logprobs={logprobs} />
      ) : (
        <>
          {/* Line 1: the user's message, read-only context (spec 2.2). */}
          <UserLine prompt={puzzle.prompt} />

          {/*
            Line 2: the assistant line (spec 2.2) — also the element that
            shakes. One .display-line div per visual line of the player's
            input; each token renders as a colored cell (AssistantLine).
          */}
          <AssistantLine puzzle={puzzle} lines={lines} showCursor ref={lineRef} />

          {/* The n/N counter (spec 2.2), right-aligned under the line. */}
          <div className="counter">
            {tokens.length}/{puzzle.expected_num_tokens}
          </div>
        </>
      )}

      {/* A plain error line under everything (spec 5.2). */}
      {error && <div className="error">{error}</div>}

      {/*
        The real input element: invisible, but focused, so native editing
        works unmodified. Value is controlled from state; `readOnly` locks
        it while scoring (spec 2.2) — and it is blurred in the reveal.
      */}
      <textarea
        ref={textareaRef}
        className="hidden-input"
        value={input}
        readOnly={phase !== "playing"}
        onChange={(event) => setInput(event.target.value)}
        onKeyDown={onKeyDown}
        aria-label="your answer"
        spellCheck={false}
      />
    </div>
  );
}