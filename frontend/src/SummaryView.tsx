// The summary screen (spec 2.4, plan 3.7): one row per challenge with its
// score and one emoji square per token, then the grand total.

// Per-token emoji square (spec 2.4):
//   🟩  lp >= -1
//   🟨  -4 <= lp < -1
//   🟥  lp < -4
function emojiFor(logprob: number): string {
  if (logprob >= -1) return "🟩";
  if (logprob >= -4) return "🟨";
  return "🟥";
}

// A challenge's score is the sum of its token logprobs (spec 2.3).
function sum(logprobs: number[]): number {
  return logprobs.reduce((a, b) => a + b, 0);
}

// `results[i]` is challenge i's logprobs, in play order (spec 2.4:
// "#idx"). The grand total is the sum over every logprob in the set.
export default function SummaryView({ results }: { results: number[][] }) {
  const total = results.flatMap((logprobs) => logprobs).reduce((a, b) => a + b, 0);

  return (
    <div className="summary">
      {results.map((logprobs, i) => (
        // Grid columns line the rows up: #idx | score | emoji squares.
        <div className="summary-row" key={i}>
          <span>#{i + 1}</span>
          <span>{sum(logprobs).toFixed(2)}</span>
          <span className="summary-emoji">
            {logprobs.map(emojiFor).join("")}
          </span>
        </div>
      ))}
      <div className="summary-total">Total: {total.toFixed(2)}</div>
    </div>
  );
}