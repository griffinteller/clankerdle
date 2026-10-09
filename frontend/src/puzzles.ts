// Puzzle data (spec section 3, plan 3.1).
//
// The puzzles are a static JSON file bundled into the frontend at build
// time — the server never sees them (spec section 4). This module is the
// single place that knows the file's shape: it types it and validates it
// once, so a malformed puzzle file fails with a clear message instead of
// crashing the game halfway through a challenge.

import puzzlesJson from "./puzzles.json";

// One challenge (spec section 3). `expected_num_tokens` is hand-chosen —
// typically the token count of the "obvious" continuation (plan 4).
export type Puzzle = {
  id: string;
  prompt: string;
  assistant_prefix: string;
  expected_num_tokens: number;
};

// The puzzle set, bundled at build time. Validated once here at module
// scope — this is static data that ships with the app, so this is
// effectively a build-time check: a malformed file throws on page load
// with a clear message in the console. Played in file order (spec 3).
export const PUZZLES: Puzzle[] = loadPuzzles();

// Read + validate puzzles.json. Every field is checked because a bad
// `expected_num_tokens` (say, a float or a string) would otherwise only
// blow up inside a fetch far from the cause.
function loadPuzzles(): Puzzle[] {
  if (!Array.isArray(puzzlesJson)) {
    throw new Error("puzzles.json: expected a JSON array of puzzles");
  }

  const ids = new Set<string>();
  return puzzlesJson.map((puzzle, index) => {
    const where = `puzzles.json entry ${index}`;
    if (typeof puzzle !== "object" || puzzle === null) {
      throw new Error(`${where}: not an object`);
    }
    const fields = puzzle as Record<string, unknown>;
    for (const key of ["id", "prompt", "assistant_prefix"]) {
      if (typeof fields[key] !== "string") {
        throw new Error(`${where}: "${key}" must be a string`);
      }
    }
    const n = fields.expected_num_tokens;
    if (!Number.isInteger(n) || (n as number) < 1) {
      throw new Error(`${where}: "expected_num_tokens" must be an integer >= 1`);
    }
    if (ids.has(fields.id as string)) {
      throw new Error(`${where}: duplicate id "${String(fields.id)}" (ids must be unique)`);
    }
    ids.add(fields.id as string);

    return {
      id: fields.id as string,
      prompt: fields.prompt as string,
      assistant_prefix: fields.assistant_prefix as string,
      expected_num_tokens: n as number,
    };
  });
}