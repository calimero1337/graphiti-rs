# graphiti-rs

A Rust implementation of [Graphiti](https://github.com/getzep/graphiti) — a framework for building temporally-aware knowledge graphs for AI agents.

## Features

- **Temporal knowledge graph** with episodic, entity, and community nodes
- **Neo4j driver** with connection pooling and schema management
- **Hybrid search** — BM25 + vector similarity with reciprocal rank fusion
- **LLM-powered pipeline** — entity/edge extraction, deduplication, contradiction resolution
- **Community detection** — Louvain-based clustering with LLM summarization
- **REST API** (axum) with full CRUD + search + contextualize + feedback endpoints
- **MCP server** (rmcp) with streamable HTTP transport — 10 tools for knowledge graph operations
- **Multiple LLM backends** — Claude CLI (via claude-agent-sdk), OpenAI, Anthropic direct
- **Multiple embedder backends** — HTTP (OpenAI-compatible), OpenAI native

## Quick Start

```bash
# Required: Neo4j and an LLM backend
export NEO4J_URI=bolt://localhost:7687
export NEO4J_USER=neo4j
export NEO4J_PASSWORD=your-password
export LLM_BACKEND=claude  # or openai, anthropic

cargo run --release --bin graphiti-server
```

The server listens on `0.0.0.0:8080` by default.

## REST Endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | `/v1/episodes` | Ingest an episode |
| GET | `/v1/episodes` | List episodes |
| DELETE | `/v1/episodes/{uuid}` | Remove an episode |
| POST | `/v1/search` | Hybrid search |
| POST | `/v1/contextualize` | Context retrieval with 1-hop expansion |
| POST | `/v1/outcomes` | Record feedback |
| POST | `/v1/timeline` | Entity timeline |
| POST | `/v1/communities/build` | Trigger community detection |
| GET | `/v1/token-usage` | LLM token stats |
| GET | `/health` | Liveness probe |
| GET | `/ready` | Readiness probe |

## MCP Tools

The MCP endpoint at `/mcp` exposes: `add_episode`, `search`, `get_entity`, `list_episodes`, `build_communities`, `contextualize`, `record_outcome`, `get_timeline`, `remember`, `get_token_usage`.

## Kubernetes

Manifests in `k8s/`:

```bash
kubectl apply -f k8s/
```

Deployed in the `claude` namespace on NodePort 30239. Requires Neo4j (manifests in `k8s/neo4j/`).

## Docker

```bash
DOCKER_BUILDKIT=1 docker build \
  --secret id=git_credentials,src=$HOME/.git-credentials \
  -t graphiti-rs:latest .
```

Build secrets are needed for the private `claude-agent-sdk-rs` dependency.

## License

Apache-2.0
