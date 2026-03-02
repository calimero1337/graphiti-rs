# ─── Stage 1: Builder ────────────────────────────────────────────────────────
FROM rust:1.93-slim AS builder

# System deps needed to compile native dependencies (OpenSSL, etc.)
RUN apt-get update && apt-get install -y --no-install-recommends \
    pkg-config \
    libssl-dev \
    ca-certificates \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

# Copy manifests first to cache dependency compilation separately from source.
COPY Cargo.toml Cargo.lock ./

# Create dummy source and bench files so `cargo build` can resolve all targets.
RUN mkdir -p src/bin benches && \
    echo 'fn main() {}' > src/bin/server.rs && \
    echo '' > src/lib.rs && \
    echo 'fn main() {}' > benches/search.rs && \
    echo 'fn main() {}' > benches/similarity.rs

# Fetch dependencies (uses git credentials for private repos).
RUN --mount=type=secret,id=git_credentials,target=/root/.git-credentials \
    git config --global credential.helper 'store' && \
    CARGO_NET_GIT_FETCH_WITH_CLI=true \
    cargo build --release --bin graphiti-server 2>/dev/null; \
    # Remove the dummy artifacts AND final binary so the real build picks up actual sources.
    rm -f target/release/graphiti-server target/release/deps/graphiti_rs* target/release/deps/graphiti-server* target/release/deps/libgraphiti_rs*

# Copy real source tree.
COPY src ./src

# Build the production binary.
RUN --mount=type=secret,id=git_credentials,target=/root/.git-credentials \
    git config --global credential.helper 'store' && \
    CARGO_NET_GIT_FETCH_WITH_CLI=true \
    cargo build --release --bin graphiti-server

# ─── Stage 2: Runtime ────────────────────────────────────────────────────────
FROM ubuntu:24.04 AS runtime

ENV DEBIAN_FRONTEND=noninteractive

# System dependencies: Node.js (for claude CLI), git, curl, OpenSSL
RUN apt-get update && apt-get install -y --no-install-recommends \
    nodejs npm \
    git curl ca-certificates \
    libssl3t64 \
    && rm -rf /var/lib/apt/lists/*

# Claude Code CLI (installed globally via npm)
RUN npm install -g @anthropic-ai/claude-code && npm cache clean --force

# Create a non-root user for the service.
RUN groupadd --gid 10001 graphiti && \
    useradd --uid 10001 --gid graphiti --shell /bin/bash --create-home graphiti

WORKDIR /app

COPY --from=builder /build/target/release/graphiti-server /app/graphiti-server

# Drop privileges.
USER graphiti

# The server binds to this port (matches LISTEN_ADDR env var).
EXPOSE 8080

ENTRYPOINT ["/app/graphiti-server"]
