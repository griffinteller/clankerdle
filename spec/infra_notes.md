# Plan Notes

## Architecture

The frontend (`web/`) will be a vite react-ts project. 

The main backend `host-app/` will be a rust executable using actix-web to expose the game api. `ConfigRepository` and `PuzzleRepository` are interfaces within this rust crate to access the database (everything is in one PostgreSQL database). This should have a dockerfile attached to build it into an image for fly. We will use the cargo chef docker template:

```dockerfile
FROM lukemathwalker/cargo-chef:latest-rust-1 AS chef
WORKDIR /app

FROM chef AS planner
COPY . .
RUN cargo chef prepare --recipe-path recipe.json

FROM chef AS builder 
COPY --from=planner /app/recipe.json recipe.json
# Build dependencies - this is the caching Docker layer!
RUN cargo chef cook --release --recipe-path recipe.json
# Build application
COPY . .
RUN cargo build --release --bin host-app

# We do not need the Rust toolchain to run the binary!
FROM debian:trixie-slim AS runtime
WORKDIR /app
COPY --from=builder /app/target/release/host-app /usr/local/bin
ENTRYPOINT ["/usr/local/bin/host-app"] 
```

`host-app/` should also have a `fly.toml` that exposes an https interface.

The inference backend `inference-app/` will be a private fly app (no https, deployed with flycast), but otherwise structured similar to `host-app`. It exposes the `BackendInterface` as its API. It should start its own llama.cpp server locally, and the model should be baked into the docker image.

The database will be hosted on Render, but this should be irrelevant. When needed, there will be an environment variable accessible called `DB_URI` which has the full postgres uri. 

All rust web apps should use actix-web for handling, reqwest for fetching, and sqlx (with tls-rustls-ring-webpki) for database operations. I have verified that these work with the hosting setup.

## Hosting

**Frontend**: Render (Static Site)

**host-app**: Fly app

**inference-app**: Fly app

**Database**: Render (Managed PostgreSQL)