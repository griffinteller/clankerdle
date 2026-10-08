// Central place for environment-driven configuration.
//
// VITE_INFERENCE_URL is the base URL of the inference app
// (spec section 4.1). It defaults to http://localhost:8080 so that
// `npm run dev` works out of the box with a locally running server.
//
// To point at a different server, set it in frontend/.env:
//   VITE_INFERENCE_URL=http://localhost:8080

export const INFERENCE_URL: string =
  import.meta.env.VITE_INFERENCE_URL ?? "http://localhost:8080";
