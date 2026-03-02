# ─── Stage 1: Builder ────────────────────────────────────────────────────────
FROM rust:1.93-slim AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    pkg-config \
    libssl-dev \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

COPY Cargo.toml Cargo.lock ./

RUN mkdir -p src/bin benches && \
    echo 'fn main() {}' > src/bin/server.rs && \
    echo '' > src/lib.rs && \
    echo 'fn main() {}' > benches/search.rs && \
    echo 'fn main() {}' > benches/similarity.rs

RUN cargo build --release --bin graphiti-server 2>/dev/null; \
    rm -f target/release/graphiti-server target/release/deps/graphiti_rs* target/release/deps/graphiti-server* target/release/deps/libgraphiti_rs*

COPY src ./src

RUN cargo build --release --bin graphiti-server

# ─── Stage 2: Runtime ────────────────────────────────────────────────────────
FROM debian:trixie-slim AS runtime

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates libssl3t64 && rm -rf /var/lib/apt/lists/*

RUN groupadd --gid 10001 graphiti && \
    useradd --uid 10001 --gid graphiti --shell /bin/bash --create-home graphiti

WORKDIR /app

COPY --from=builder /build/target/release/graphiti-server /app/graphiti-server

USER graphiti
EXPOSE 8080
ENTRYPOINT ["/app/graphiti-server"]
