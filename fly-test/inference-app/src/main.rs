use actix_web::{App, HttpResponse, HttpServer, Responder, post};

#[post("/")]
async fn index(body: String) -> impl Responder {
    HttpResponse::Ok().body(body.chars().rev().collect::<String>())
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