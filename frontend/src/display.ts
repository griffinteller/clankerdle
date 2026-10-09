// Pure helpers that turn tokens into something renderable (plan 3.4/3.6).
//
// The player may type anything (spec 2.2), including newlines — so token
// texts can contain newlines, and a single token can even straddle a line
// break (e.g. a "\n\n" token). To draw one bar under each token (spec 2.3),
// we split the token stream into "display lines": every newline inside a
// token text ends its display line. Each token then appears as one or more
// *fragments* — the pieces of its text that sit on each display line — and
// its bar hangs under its first fragment.
//
// This module is plain TS with no Vite-specific imports (same style as
// tokenizer.ts) and pure, so it is trivial to reason about and test.

import type { Token } from "./tokenizer";

// The glyph shown before every line break (spec 2.2: "a newline token shows
// as a ↵ glyph before the break").
export const NEWLINE_GLYPH = "↵";

// One piece of one token's text, on one display line.
export type Fragment = {
  // Which token this fragment belongs to (index into the token array).
  tokenIndex: number;
  // The token's text on this display line (never contains a newline).
  text: string;
  // True when this fragment is followed by a line break (i.e. the token's
  // text had a newline here) — the UI appends the ↵ glyph.
  newline: boolean;
  // True for the token's first fragment — the one its bar hangs under.
  first: boolean;
};

// Split tokens into display lines of fragments (see the module comment).
// A newline, \r\n or lone \r inside a token each start a new display line.
// With no tokens this returns one empty line — the UI renders that single
// (empty) line so the cursor still has a place to sit.
export function displayLines(tokens: Token[]): Fragment[][] {
  const lines: Fragment[][] = [[]];
  tokens.forEach((token, tokenIndex) => {
    const parts = token.text.split(/\r\n|\r|\n/);
    parts.forEach((text, partIndex) => {
      // Every part after the first follows a newline, so it starts a new
      // display line.
      if (partIndex > 0) lines.push([]);
      lines[lines.length - 1].push({
        tokenIndex,
        text,
        newline: partIndex < parts.length - 1,
        first: partIndex === 0,
      });
    });
  });
  return lines;
}

// The text a fragment's span shows: its own text, plus the ↵ glyph when a
// line break follows it (spec 2.2).
export function fragmentText(fragment: Fragment): string {
  return fragment.text + (fragment.newline ? NEWLINE_GLYPH : "");
}