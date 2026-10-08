//! Rendering the chat context with the GGUF's own embedded template.
//!
//! The spec (4.2) says the model sees Gemma's *real* chat template — the one
//! embedded in the GGUF, not a hard-coded one. Gemma 4's template is written
//! in full Jinja (macros, `namespace()`, `dictsort`, `.get()`, …), and
//! llama.cpp's built-in `llama_chat_apply_template` (which llama-cpp-2 wraps)
//! only implements a small Jinja subset — it fails with an FFI error on this
//! template.
//!
//! So we render the *same template string* with minijinja instead, which is
//! the engine llama.cpp's own server and HF's Rust tokenizers use for chat
//! templates. Same template, same output as real inference.
//!
//! One wrinkle: the template calls `message.get('reasoning')` (Jinja2 dict
//! method), which plain minijinja maps don't have — so messages are passed as
//! a small custom Object below that supports `.get()` like a Python dict.

use llama_cpp_2::model::LlamaModel;
use minijinja::value::{Object, Value};
use minijinja::{Environment, Error, ErrorKind};

/// Holds the GGUF's chat template, compiled once.
///
/// minijinja's `Environment` is Send+Sync, so the server can share one of
/// these across all requests without a mutex.
pub struct ChatRenderer {
    env: Environment<'static>,
}

impl ChatRenderer {
    /// Read the template out of the model's GGUF metadata and compile it.
    /// Fails at startup (and in tests) if the model has no usable template.
    pub fn new(model: &LlamaModel) -> Result<Self, String> {
        // `None` = the model's default chat template.
        let template = model
            .chat_template(None)
            .map_err(|e| format!("the GGUF has no chat template: {e}"))?
            .to_string()
            .map_err(|e| format!("the chat template is not valid UTF-8: {e}"))?;

        let mut env = Environment::new();
        env.add_template_owned("chat", template)
            .map_err(|e| format!("the chat template does not compile: {e}"))?;

        Ok(ChatRenderer { env })
    }

    /// Render a single user message, with the generation prompt enabled, so
    /// the string ends exactly at the start of the model's turn:
    ///
    /// `<bos><|turn>user\n{prompt}<turn|>\n<|turn>model\n`
    ///
    /// (for this Gemma 4 GGUF — whatever the template says, we follow it).
    ///
    /// `bos_token`/`eos_token` are taken from the model's vocab so the
    /// renderer never hard-codes token spellings.
    pub fn render_user_turn(&self, model: &LlamaModel, prompt: &str) -> Result<String, String> {
        let vocab = model.vocab();
        // vocab.text() gives the human-readable spelling of a special token,
        // e.g. "<bos>". Fall back to "" — templates guard with `if bos_token`.
        let bos = vocab.text(vocab.bos()).map(|s| s.to_string_lossy().to_string());
        let eos = vocab.text(vocab.eos()).map(|s| s.to_string_lossy().to_string());

        let message = Value::from_object(ChatMessage {
            role: "user".to_string(),
            content: prompt.to_string(),
        });

        let rendered = self
            .env
            .get_template("chat")
            .expect("template was added in new()")
            .render(minijinja::context! {
                messages => vec![message],
                add_generation_prompt => true,
                bos_token => bos,
                eos_token => eos,
            })
            .map_err(|e| format!("rendering the chat template failed: {e}"))?;

        Ok(rendered)
    }
}

/// One chat message, exposed to the template as a Jinja2-style dict:
/// `message.role` / `message['content']` / `message.get('reasoning')`.
///
/// Only the fields Clankerdle sends are ever set, so every other lookup
/// (tool_calls, reasoning, …) correctly behaves as "missing".
#[derive(Debug)]
struct ChatMessage {
    role: String,
    content: String,
}

impl ChatMessage {
    fn field(&self, key: &str) -> Option<Value> {
        match key {
            "role" => Some(Value::from(self.role.as_str())),
            "content" => Some(Value::from(self.content.as_str())),
            _ => None, // unknown key -> undefined, which the template treats as missing
        }
    }
}

impl Object for ChatMessage {
    // Attribute lookup: message['role'], message['content'], … The Value-keyed
    // version is what subscript access uses; keep both pointed at `field`.
    fn get_value(self: &std::sync::Arc<Self>, key: &Value) -> Option<Value> {
        self.field(&key.as_str()?.to_string())
    }

    fn get_value_by_str(self: &std::sync::Arc<Self>, key: &str) -> Option<Value> {
        self.field(key)
    }

    // Jinja2 dicts also have .get(key); the Gemma template uses it
    // (`message.get('reasoning')` etc.). Mirror that behavior exactly.
    fn call_method(
        self: &std::sync::Arc<Self>,
        _state: &minijinja::State,
        method: &str,
        args: &[Value],
    ) -> Result<Value, Error> {
        if method == "get" {
            if let Some(key) = args.first().and_then(Value::as_str) {
                return Ok(self.field(&key).unwrap_or(Value::UNDEFINED));
            }
        }
        Err(Error::new(
            ErrorKind::UnknownMethod,
            format!("ChatMessage has no method named {method}"),
        ))
    }
}
