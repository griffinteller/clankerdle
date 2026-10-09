// Central place for configuration (mostly environment-driven).
//
// VITE_INFERENCE_URL is the base URL of the inference app
// (spec section 4.1). It defaults to http://localhost:8080 so that
// `npm run dev` works out of the box with a locally running server.
//
// To point at a different server, set it in frontend/.env:
//   VITE_INFERENCE_URL=http://localhost:8080

export const INFERENCE_URL: string =
  import.meta.env.VITE_INFERENCE_URL ?? "http://localhost:8080";

// The name of the game (spec 2.1: "no branding beyond a small [name]
// title"). This is the ONE variable behind the game's name everywhere:
//
//   - the tab/page title (main.tsx sets document.title from it)
//   - the heading on every screen (App.tsx's Shell)
//
// Change it here (or override it without touching code via
// VITE_GAME_NAME in frontend/.env) and the whole frontend follows.
export const GAME_NAME: string = import.meta.env.VITE_GAME_NAME ?? "slopdle";