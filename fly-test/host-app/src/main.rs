use std::env;

use actix_web::{App, HttpResponseBuilder, HttpServer, Responder, get, http::{StatusCode, header}, mime, web};
use chrono::Local;
use serde::Deserialize;
use serde_json::json;

#[derive(Deserialize)]
struct Info {
    info: String
}

#[get("/")]
async fn index(info: web::Query<Info>) -> Result<impl Responder, Box<dyn std::error::Error>> {
    let client = reqwest::Client::new();
    let res = client
        .post(env::var("INFERENCE_ENDPOINT")?)
        .body(info.info.clone())
        .send()
        .await?;

    let time = Local::now().to_rfc2822();

    let bytes = res.bytes().await?;
    let s = String::from_utf8(bytes.to_vec())?;

    Ok(HttpResponseBuilder::new(StatusCode::OK)
        .insert_header((header::CONTENT_TYPE, mime::APPLICATION_JSON))
        .insert_header((header::ACCESS_CONTROL_ALLOW_ORIGIN, "*"))
        .insert_header((header::VARY, "Origin"))
        .message_body(json!({
            "info": s,
            "time": time,
        }).to_string())?)
}


#[actix_web::main]
async fn main() -> std::io::Result<()> {
    HttpServer::new(|| {
        App::new()
            .service(index)
    })
    .bind(("0.0.0.0", 8080))?
    .run()
    .await?;

    Ok(())
}