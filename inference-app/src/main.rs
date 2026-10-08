//! Clankerdle inference app — HTTP server entry point.
//!
//! Phase 0 scope (see spec/plan.md):
//!   - read PORT / CLANKERDLE_MODEL_DIR from the environment
//!   - validate the model directory (exactly one *.gguf + tokenizer.json)
//!   - load the model once at startup
//!   - start an actix-web server on 127.0.0.1 with permissive CORS
//!
//! The actual endpoints (`GET /tokenizer`, `POST /score`) are Phase 1 and
//! will be registered inside `configure_routes` below.

mod model;

use std::path::PathBuf;
use std::sync::Mutex;

use actix_cors::Cors;
use actix_web::{web, App, HttpResponse, HttpServer};
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::model::LlamaModel;

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
/// The spec (4.2, Concurrency) says a single model is shared behind a mutex
/// and requests are scored one at a time, which is fine for local use.
#[derive(Clone)]
struct AppState {
    /// The loaded model. Handlers lock this during scoring.
    model: web::Data<Mutex<LlamaModel>>,
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

/// Route registration. Phase 1 will add `GET /tokenizer` and `POST /score`
/// here; for now the server just runs and answers 404 to everything, which
/// is enough to smoke-test startup.
fn configure_routes(_cfg: &mut web::ServiceConfig) {}

/// Temporary root handler for smoke-testing: proves the server is up and the
/// model made it into the shared state. Removed when the real routes land.
async fn index(state: web::Data<AppState>) -> HttpResponse {
    // Lock the mutex just to prove the model is reachable from a handler.
    let n_vocab = state
        .model
        .lock()
        .expect("model mutex poisoned")
        .n_vocab();
    HttpResponse::Ok().json(serde_json::json!({ "status": "ok", "n_vocab": n_vocab }))
}

#[actix_web::main]
async fn main() -> std::io::Result<()> {
    let config = Config::from_env();

    // Validate the model directory before touching llama.cpp, so a missing
    // file produces a clear message instead of a crash inside the library.
    let paths = model::validate_model_dir(&config.model_dir).unwrap_or_else(|e| {
        eprintln!("error: {e}");
        std::process::exit(1);
    });

    // Initialize llama.cpp exactly once. The backend handle is an empty
    // "proof of initialization" token; we keep it alive for the whole
    // program, but the model itself does not borrow it.
    let backend = LlamaBackend::init().unwrap_or_else(|e| {
        eprintln!("error: failed to initialize llama backend: {e}");
        std::process::exit(1);
    });

    // Load the model once, before the server starts accepting requests.
    let llama_model = model::load_model(&backend, &paths.gguf).unwrap_or_else(|e| {
        eprintln!("error: {e}");
        std::process::exit(1);
    });
    println!(
        "model loaded ({} context tokens trained, {} vocab size)",
        llama_model.n_ctx_train(),
        llama_model.n_vocab()
    );

    // Shared state: one model behind one mutex (spec 4.2, Concurrency).
    let model = web::Data::new(Mutex::new(llama_model));
    let state = web::Data::new(AppState { model: model.clone() });

    let bind_addr = ("127.0.0.1", config.port);
    println!("listening on http://{}:{}", bind_addr.0, bind_addr.1);

    HttpServer::new(move || {
        App::new()
            .app_data(state.clone())
            .wrap(cors())
            .route("/", web::get().to(index))
            .configure(configure_routes)
    })
    .bind(bind_addr)?
    .run()
    .await
}
