// Phase 2: the app proves out spec 4.1 — load the tokenizer from the
// inference app before anything else. Phase 3 replaces this screen with the
// actual game UI (spec section 2).
//
//   - while the tokenizer is loading: "loading…"
//   - if the fetch fails: a plain error message
//   - once ready: the game would start here (phase 3)

import { useEffect, useState } from "react";
import { INFERENCE_URL } from "./config";
import { loadTokenizer } from "./tokenizer";
import "./App.css";

// Status of the tokenizer load. This is all the app can do until the
// tokenizer arrives, so it is the entire phase-2 UI.
type Status =
  | { kind: "loading" }
  | { kind: "error"; message: string }
  | { kind: "ready" };

function App() {
  const [status, setStatus] = useState<Status>({ kind: "loading" });

  // Fetch tokenizer.json from the inference app once. In dev, React's
  // StrictMode mounts twice, so this may run twice — harmless (the tokenizer
  // is small and stateless; phase 3 will keep the loaded tokenizer in
  // component state instead).
  useEffect(() => {
    let cancelled = false;
    loadTokenizer(INFERENCE_URL)
      .then(() => {
        if (!cancelled) setStatus({ kind: "ready" });
      })
      .catch((err: unknown) => {
        // A plain error message (spec 4.1): no retry, no stack.
        const message = err instanceof Error ? err.message : String(err);
        if (!cancelled) setStatus({ kind: "error", message });
      });
    return () => {
      cancelled = true;
    };
  }, []);

  return (
    <main>
      <h1>Clankerdle</h1>
      {status.kind === "loading" && <p>loading…</p>}
      {status.kind === "error" && <p className="error">{status.message}</p>}
      {status.kind === "ready" && <p>tokenizer ready — phase 3 adds the game.</p>}
    </main>
  );
}

export default App;