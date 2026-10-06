use std::env;

use actix_web::{App, Error, HttpResponseBuilder, HttpServer, Responder, ResponseError, get, http::{StatusCode, header}, mime, post, web};
use chrono::Local;
use serde::{Deserialize, Serialize};
use serde_json::json;
use sqlx::{PgPool, Row, postgres::{PgPoolOptions, PgRow}};

#[derive(Deserialize, Serialize)]
struct Info {
    info: String,
    time: String,
}

#[get("/info")]
async fn get_info(
    web::ThinData(db_pool): web::ThinData<PgPool>
) -> Result<impl Responder, Box<dyn std::error::Error>> {
    // get all rows of table 'info'

    let rows = sqlx::query("SELECT info, time FROM info ORDER BY time DESC")
        .fetch_all(&db_pool)
        .await?
        .iter()
        .map(|row| Ok(Info {
            info: row.try_get("info")?,
            time: row.try_get("time")?,
        }))
        .collect::<Result<Vec<Info>, sqlx::Error>>()?;

    Ok(HttpResponseBuilder::new(StatusCode::OK)
        .insert_header((header::CONTENT_TYPE, mime::APPLICATION_JSON))
        .insert_header((header::ACCESS_CONTROL_ALLOW_ORIGIN, "*"))
        .insert_header((header::VARY, "Origin"))
        .message_body(json!(rows).to_string())?)
}

#[post("/info")]
async fn post_info(
    web::ThinData(db_pool): web::ThinData<PgPool>,
    body: String,
) -> Result<impl Responder, Box<dyn std::error::Error>> {
    let time = Local::now().to_rfc2822();

    sqlx::query("INSERT INTO info (info, time) VALUES ($1, $2)")
        .bind(&body)
        .bind(&time)
        .execute(&db_pool)
        .await?;

    Ok(HttpResponseBuilder::new(StatusCode::OK)
        .insert_header((header::CONTENT_TYPE, mime::APPLICATION_JSON))
        .insert_header((header::ACCESS_CONTROL_ALLOW_ORIGIN, "*"))
        .insert_header((header::VARY, "Origin"))
        .message_body(json!(Info {
            info: body,
            time: time,
        }).to_string())?)
}


#[actix_web::main]
async fn main() -> anyhow::Result<()> {
    let db_pool = PgPoolOptions::new()
        .max_connections(5)
        .connect(&env::var("DB_URI")?)
        .await?;

    let db_pool_clone = db_pool.clone();
    HttpServer::new(move || {
        App::new()
            .app_data(web::ThinData(db_pool_clone.clone()))
            .service(get_info)
            .service(post_info)
    })
    .bind(("0.0.0.0", 8080))?
    .run()
    .await?;

    db_pool
        .close()
        .await;

    Ok(())
}