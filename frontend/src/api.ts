// Client for the inference app's `POST /score` (spec section 5).
//
// (`GET /tokenizer` lives in tokenizer.ts.) Everything here mirrors the
// server's JSON exactly: the request body, the serde-tagged response
// variants, and the `{ "error": string }` error shape.

// `POST /score` request body (spec 5.2). All four fields come from the
// puzzle and the player's input.
export type ScoreRequest = {
  prompt: string;
  assistant_prefix: string;
  text: string;
  expected_num_tokens: number;
};

// `POST /score` response body (spec 5.2), serde's internally-tagged
// representation: `{"type":"Logprobs","logprobs":[...]}` or
// `{"type":"UnexpectedNumTokens",...}`.
export type ScoreResponse =
  | { type: "Logprobs"; logprobs: number[] }
  | {
      type: "UnexpectedNumTokens";
      expected_num_tokens: number;
      actual_num_tokens: number;
    };

// Send one scoring request. Throws on any 4xx/5xx (the caller shows the
// message as a plain error line and keeps the input editable, spec 2.2).
export async function postScore(baseUrl: string, request: ScoreRequest): Promise<ScoreResponse> {
  const response = await fetch(`${baseUrl}/score`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(request),
  });

  // 4xx/5xx: the server promises `{ "error": string }` (spec 5.2). Use
  // that message if we can read it; otherwise fall back to the status.
  if (!response.ok) {
    let message = `HTTP ${response.status}`;
    try {
      const body: unknown = await response.json();
      if (typeof body === "object" && body !== null) {
        const error = (body as { error?: unknown }).error;
        if (typeof error === "string") message = error;
      }
    } catch {
      // Not a JSON body — keep the HTTP status as the message.
    }
    throw new Error(`POST /score failed: ${message}`);
  }

  return parseScoreResponse(await response.json());
}

// Validate a 200 body into a ScoreResponse. The server is our own code,
// so this is not security — it just turns "the server sent something
// unexpected" into a clear error instead of a crash mid-render.
function parseScoreResponse(body: unknown): ScoreResponse {
  if (typeof body !== "object" || body === null) {
    throw new Error("POST /score returned a non-object body");
  }
  const fields = body as Record<string, unknown>;

  if (fields.type === "Logprobs") {
    const logprobs = fields.logprobs;
    if (
      !Array.isArray(logprobs) ||
      !logprobs.every((lp) => typeof lp === "number" && Number.isFinite(lp))
    ) {
      throw new Error('POST /score returned "Logprobs" without a finite number[] "logprobs"');
    }
    return { type: "Logprobs", logprobs };
  }

  if (fields.type === "UnexpectedNumTokens") {
    const expected = fields.expected_num_tokens;
    const actual = fields.actual_num_tokens;
    if (typeof expected !== "number" || typeof actual !== "number") {
      throw new Error('POST /score returned "UnexpectedNumTokens" without number fields');
    }
    return { type: "UnexpectedNumTokens", expected_num_tokens: expected, actual_num_tokens: actual };
  }

  throw new Error(`POST /score returned an unknown response type: ${String(fields.type)}`);
}