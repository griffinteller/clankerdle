// The reveal screen (spec 2.3): the challenge with its accepted answer,
// one hanging bar + printed logprob per token, the challenge score, and
// the continue hint.
//
// This is a pure function of (puzzle, tokens, logprobs) — no state, no
// listeners — for two reasons:
//
//   1. ChallengeView swaps to it when the submission is accepted.
//   2. App re-renders it as a static snapshot of the *outgoing* challenge
//      during the pseudo-scroll, so the old puzzle animates out exactly as
//      it looked (App holds the submission; see App.tsx).
//
// The bars/numbers/score animate in (staggered grow, fades — see
// AssistantLine.tsx and the `.bar`/`.lp` rules in App.css).

import { useMemo } from "react";
import { displayLines } from "./display";
import { AssistantLine, BAR_GROW_MS, BAR_STAGGER_MS, UserLine } from "./AssistantLine";
import type { Puzzle } from "./puzzles";
import type { Token } from "./tokenizer";

export default function RevealView({
  puzzle,
  tokens,
  logprobs,
}: {
  puzzle: Puzzle;
  // The submitted input, tokenized (same rules as the live counter).
  tokens: Token[];
  // One logprob per token, from POST /score (spec 5.2).
  logprobs: number[];
}) {
  const lines = useMemo(() => displayLines(tokens), [tokens]);

  // The challenge score is the sum of the token logprobs (spec 2.3).
  const score = logprobs.reduce((a, b) => a + b, 0);

  // The score line and hint wait for the bars to finish their stagger:
  // the last bar starts at (n-1) * BAR_STAGGER_MS and grows for
  // BAR_GROW_MS — add a beat. Capped so many-token puzzles don't feel slow.
  const afterBarsMs = Math.min(
    750,
    (logprobs.length - 1) * BAR_STAGGER_MS + BAR_GROW_MS + 150,
  );

  return (
    <>
      <UserLine prompt={puzzle.prompt} />
      <AssistantLine puzzle={puzzle} lines={lines} logprobs={logprobs} />
      <div className="score-line" style={{ animationDelay: `${afterBarsMs}ms` }}>
        Score: {score.toFixed(2)}
      </div>
      <div className="hint" style={{ animationDelay: `${afterBarsMs + 80}ms` }}>
        press enter to continue...
      </div>
    </>
  );
}