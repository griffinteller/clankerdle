// The shared presentational pieces of a challenge screen (plan 3.3):
//
//   - `UserLine`       — "User: <prompt>", line 1 of the spec 2.2 mockup
//   - `AssistantLine`  — "Assistant: <assistant_prefix>" + the player's
//                        tokens as colored cells, line 2: optionally with
//                        the blinking cursor (playing), and optionally with
//                        a bar + printed logprob hanging under each token
//                        (reveal, spec 2.3)
//
// One component serves both the live challenge (ChallengeView) and the
// outgoing snapshot of the pseudo-scroll (App renders it via RevealView) —
// the transition is just a re-render of the same screen, so the old puzzle
// fades out *exactly as it looked*.

import type { Ref } from "react";
import { fragmentText, type Fragment } from "./display";
import type { Puzzle } from "./puzzles";

// ---- Bar geometry & timing (spec 2.3, polished) ----
//
// A bar's height is proportional to |logprob|, clamped so a logprob of -15
// (or worse) fills BAR_MAX_PX (spec 2.3). The polish: each bar *grows*
// downward from the shared baseline with a per-token stagger
// (BAR_STAGGER_MS), and its number fades in after it — see TokenCell and
// the `.bar`/`.lp` rules in App.css.

// Delay from one bar to the next (also used by RevealView to time the
// score line after the last bar finishes).
export const BAR_STAGGER_MS = 30;
// How long one bar takes to grow, once its turn comes (matches the
// `bar-grow` animation in App.css).
export const BAR_GROW_MS = 280;
// |logprob| at which a bar is full-length (spec 2.3: clamp at -15).
const BAR_CLAMP = 15;
// Full-length bar height in pixels.
const BAR_MAX_PX = 64;

// How many pale colors the token palette cycles through (spec 2.2: "a
// small fixed list of pale colors"). The colors are the .token-0 … .token-3
// classes in App.css.
const NUM_TOKEN_COLORS = 4;

// Line 1: the user's message — read-only context (spec 2.2).
export function UserLine({ prompt }: { prompt: string }) {
  return (
    <div className="context-line">
      <span className="label">User: </span>
      <span className="context-text">{prompt}</span>
    </div>
  );
}

// Line 2: the assistant line (spec 2.2/2.3). `lines` is the player's input
// split into display lines (display.ts). Pass `logprobs` in the reveal to
// draw the hanging bars (one per token, under its first fragment), or
// `showCursor` while playing. `ref` reaches the line div so ChallengeView
// can shake it on a wrong token count (spec 2.2).
export function AssistantLine({
  puzzle,
  lines,
  logprobs,
  showCursor,
  ref,
}: {
  puzzle: Puzzle;
  lines: Fragment[][];
  // One logprob per token, in order (reveal only).
  logprobs?: number[];
  // Blinking cursor after the input (playing/scoring only).
  showCursor?: boolean;
  // React 19 passes `ref` as a plain prop — no forwardRef needed.
  ref?: Ref<HTMLDivElement>;
}) {
  return (
    <div className="assistant-line" ref={ref}>
      {lines.map((line, i) => (
        <div className="display-line" key={i}>
          {i === 0 && (
            // The read-only start of the line: label + assistant_prefix.
            // Plain text — only the player's tokens get colored (spec 2.2).
            <span className="context-text">Assistant: {puzzle.assistant_prefix}</span>
          )}
          {line.map((fragment, j) => (
            <TokenCell
              key={`${i}-${j}`}
              fragment={fragment}
              // Colors and bar-stagger delays cycle by token position.
              colorIndex={fragment.tokenIndex}
              // Only the token's first fragment carries its bar; a bar is
              // drawn at all only in the reveal (spec 2.3).
              logprob={
                logprobs !== undefined && fragment.first
                  ? logprobs[fragment.tokenIndex]
                  : undefined
              }
            />
          ))}
          {showCursor && i === lines.length - 1 && (
            // The blinking cursor after the input (spec 2.2), in flow.
            <span className="cursor">▌</span>
          )}
        </div>
      ))}
    </div>
  );
}

// One cell of the assistant line: the token fragment's text on top, and in
// the reveal a bar + the printed logprob hanging below it (spec 2.3). Cells
// are inline-flex columns, so the bar starts directly under the text and
// all cells' bars share a baseline (see App.css).
function TokenCell({
  fragment,
  colorIndex,
  logprob,
}: {
  fragment: Fragment;
  // The token's index — colors cycle by token position (spec 2.2).
  colorIndex: number;
  // Undefined unless this is the revealed token's first fragment.
  logprob?: number;
}) {
  return (
    <span className="cell">
      <span className={`text token-${colorIndex % NUM_TOKEN_COLORS}`}>
        {fragmentText(fragment)}
      </span>
      {logprob !== undefined && (
        <>
          {/* The bar grows downward (transform-origin: top) once its
              stagger delay passes; the height itself is the true length. */}
          <span
            className="bar"
            style={{
              height: `${barHeightPx(logprob)}px`,
              animationDelay: `${colorIndex * BAR_STAGGER_MS}ms`,
            }}
          />
          {/* The true logprob, 2 decimal places (spec 2.3), fading in a
              beat after its bar. */}
          <span
            className="lp"
            style={{ animationDelay: `${colorIndex * BAR_STAGGER_MS + BAR_GROW_MS / 2}ms` }}
          >
            {logprob.toFixed(2)}
          </span>
        </>
      )}
    </span>
  );
}

// Bar height in pixels (spec 2.3): proportional to |logprob|, clamped so a
// logprob of -15 (or worse) fills BAR_MAX_PX. `Math.max(1, …)` keeps
// near-zero logprobs visible as a 1px sliver. The printed number is always
// the true value — only the bar is clamped.
function barHeightPx(logprob: number): number {
  return Math.max(1, (Math.min(Math.abs(logprob), BAR_CLAMP) / BAR_CLAMP) * BAR_MAX_PX);
}