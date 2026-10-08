//! Clankerdle inference app — HTTP server entry point.
//!
//! Serves two endpoints (spec 5):
//!   - `GET /tokenizer`  — streams the model dir's tokenizer.json
//!   - `POST /score`     — one logprob per token of the player's text
//!
//! Startup (spec 4.2): validate the model directory (exactly one *.gguf +
//! tokenizer.json), load the model once, create one inference context, and
//! share both behind a mutex so requests are scored one at a time.

mod chat;
mod model;
mod score;

use std::num::NonZeroU32;
use std::path::PathBuf;
use std::sync::Mutex;

use actix_cors::Cors;
use actix_web::web::{self, Bytes};
use actix_web::{App, HttpResponse, HttpServer};
use llama_cpp_2::context::params::LlamaContextParams;
use llama_cpp_2::context::LlamaContext;
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::model::LlamaModel;

/// Size of the KV cache / batch, in tokens. Kept in one place (score.rs) so
/// the server and the tests construct identical contexts.

/// Server configuration, read once from the environment at startup.
struct Config {
    /// Port to listen on. Overridable with PORT. Default 8080 (spec 4.2).
    port: u16,
    /// Directory containing the GGUF and tokenizer.json.
    /// Overridable with CLANKERDLE_MODEL_DIR. Default ./models (spec 4.2).
    model_dir: PathBuf,
}

impl Config {
    fn from_env() -> Self {
        let port = std::env::var("PORT")
            .ok()
            .and_then(|p| p.parse().ok())
            .unwrap_or(8080);

        let model_dir = std::env::var("CLANKERDLE_MODEL_DIR")
            .map(PathBuf::from)
            .unwrap_or_else(|_| PathBuf::from("./models"));

        Config { port, model_dir }
    }
}

/// Everything the request handlers need.
///
/// The model and its context are loaded once at startup and shared behind a
/// mutex: a single model + single context, requests scored one at a time
/// (spec 4.2, Concurrency).
struct AppState {
    /// The loaded model, shared by every handler. It is deliberately leaked
    /// (see main) so it lives as long as the process — matching how the
    /// context borrows from it — which is fine for a server that never
    /// outlives its model anyway.
    model: &'static LlamaModel,
    /// Renders the GGUF's own chat template (see chat.rs). Immutable and
    /// thread-safe, so it needs no mutex. In its own Arc so handlers can take
    /// a cheap clone into the web::block closure.
    renderer: web::Data<chat::ChatRenderer>,
    /// The one inference context. Scoring locks this mutex.
    context: web::Data<Mutex<LlamaContext<'static>>>,
    /// Contents of tokenizer.json, read once at startup and served verbatim.
    tokenizer_json: Bytes,
}

impl AppState {
    /// Convenience constructor used by main: bundle the shared pieces.
    fn new(
        model: &'static LlamaModel,
        renderer: web::Data<chat::ChatRenderer>,
        context: web::Data<Mutex<LlamaContext<'static>>>,
        tokenizer_json: Bytes,
    ) -> web::Data<AppState> {
        web::Data::new(AppState {
            model,
            renderer,
            context,
            tokenizer_json,
        })
    }
}

/// Permissive CORS (spec 4.2): any origin, but only the methods and
/// headers we actually use — GET, POST and the Content-Type header.
/// Needed so the Vite dev server on localhost:5173 can call us directly.
fn cors() -> Cors {
    Cors::default()
        .allow_any_origin()
        .allowed_methods(vec!["GET", "POST"])
        .allowed_headers(vec!["Content-Type"])
}

/// `GET /tokenizer` (spec 5.1): serve tokenizer.json as-is.
async fn tokenizer(state: web::Data<AppState>) -> HttpResponse {
    HttpResponse::Ok()
        .content_type("application/json")
        .body(state.tokenizer_json.clone())
}

/// `POST /score` (spec 5.2).
async fn score(body: Bytes, state: web::Data<AppState>) -> HttpResponse {
    // Parse by hand instead of using the Json extractor so malformed input
    // gets our {"error": ...} body instead of actix's default.
    let req: score::ScoreRequest = match serde_json::from_slice(&body) {
        Ok(req) => req,
        Err(e) => return bad_request(format!("malformed JSON: {e}")),
    };

    // The spec pins expected_num_tokens >= 1 (0 would mean "no tokens to score").
    if req.expected_num_tokens < 1 {
        return bad_request("expected_num_tokens must be >= 1".to_string());
    }

    // Scoring is blocking CPU work, so it runs on actix's thread pool via
    // web::block instead of blocking an async worker. The context mutex is
    // locked inside the closure and released before we await.
    let context = state.context.clone();
    let model = state.model;
    let renderer = state.renderer.clone();
    let result = web::block(move || {
        let mut context = context.lock().expect("context mutex poisoned");
        score::score(model, &renderer, &mut context, &req)
    })
    .await;

    match result {
        // Both variants (Logprobs and UnexpectedNumTokens) are 200s.
        Ok(Ok(response)) => HttpResponse::Ok().json(response),
        Ok(Err(e)) => server_error(e),
        Err(_) => server_error("the scoring task failed to run".to_string()),
    }
}

/// 400 with the spec's error body shape.
fn bad_request(message: String) -> HttpResponse {
    HttpResponse::BadRequest().json(serde_json::json!({ "error": message }))
}

/// 500 with the spec's error body shape.
fn server_error(message: String) -> HttpResponse {
    HttpResponse::InternalServerError().json(serde_json::json!({ "error": message }))
}

#[actix_web::main]
async fn main() -> std::io::Result<()> {
    // The app runs on the CPU only (spec 4.2), so ask ggml to create zero
    // Metal devices. This skips Metal discovery at startup and avoids a
    // llama.cpp crash in Metal cleanup on some macOS versions at exit.
    std::env::set_var("GGML_METAL_DEVICES", "0");

    let config = Config::from_env();

    // Validate the model directory before touching llama.cpp, so a missing
    // file produces a clear message instead of a crash inside the library.
    let paths = model::validate_model_dir(&config.model_dir).unwrap_or_else(|e| {
        eprintln!("error: {e}");
        std::process::exit(1);
    });

    // Initialize llama.cpp exactly once. The backend handle is an empty
    // "proof of initialization" token. We leak it (and the model below) so
    // both live for the rest of the process — the inference context borrows
    // from the model, and a server never wants to drop its model anyway.
    let backend: &'static LlamaBackend = Box::leak(Box::new(
        LlamaBackend::init().unwrap_or_else(|e| {
            eprintln!("error: failed to initialize llama backend: {e}");
            std::process::exit(1);
        }),
    ));

    // Load the model once, before the server starts accepting requests.
    let llama_model: &'static LlamaModel =
        Box::leak(Box::new(model::load_model(backend, &paths.gguf).unwrap_or_else(|e| {
            eprintln!("error: {e}");
            std::process::exit(1);
        })));
    println!(
        "model loaded ({} context tokens trained, {} vocab size)",
        llama_model.n_ctx_train(),
        llama_model.n_vocab()
    );

    // One inference context, big enough for a puzzle prompt plus a guess
    // in a single batch. Larger requests fail with a 500 (spec 5.2).
    let context_params = LlamaContextParams::default()
        .with_n_ctx(NonZeroU32::new(score::N_CTX))
        .with_n_batch(score::N_CTX);
    let context = llama_model.new_context(backend, context_params).unwrap_or_else(|e| {
        eprintln!("error: failed to create inference context: {e}");
        std::process::exit(1);
    });

    // Compile the GGUF's chat template and smoke-render one message so a
    // broken template fails loudly at startup, not on the first request.
    let renderer = web::Data::new(chat::ChatRenderer::new(llama_model).unwrap_or_else(|e| {
        eprintln!("error: {e}");
        std::process::exit(1);
    }));
    if let Err(e) = renderer.render_user_turn(llama_model, "startup check") {
        eprintln!("error: the chat template does not render: {e}");
        std::process::exit(1);
    }

    // Read tokenizer.json once; GET /tokenizer serves these bytes verbatim.
    let tokenizer_json = Bytes::from(
        std::fs::read(&paths.tokenizer_json).unwrap_or_else(|e| {
            eprintln!("error: failed to read {}: {e}", paths.tokenizer_json.display());
            std::process::exit(1);
        }),
    );

    // Shared state: one model + one context behind one mutex (spec 4.2).
    let state = AppState::new(
        llama_model,
        renderer,
        web::Data::new(Mutex::new(context)),
        tokenizer_json,
    );

    let bind_addr = ("127.0.0.1", config.port);
    println!("listening on http://{}:{}", bind_addr.0, bind_addr.1);

    HttpServer::new(move || {
        App::new()
            .app_data(state.clone())
            .wrap(cors())
            .route("/tokenizer", web::get().to(tokenizer))
            .route("/score", web::post().to(score))
    })
    .bind(bind_addr)?
    .run()
    .await
}
