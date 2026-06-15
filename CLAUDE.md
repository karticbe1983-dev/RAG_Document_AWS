# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install all dependencies
pip install -r requirements.txt -r requirements-dev.txt

# Run all unit tests (no AWS credentials required)
pytest tests/ -m "not integration"

# Run a single test class or function
pytest tests/test_chunking.py::TestFixedSizeChunker -v
pytest tests/test_rag.py::TestRAGGenerator::test_generate_returns_rag_response -v

# Run integration tests (requires live AWS credentials + deployed infra)
pytest tests/ -m integration

# Lint
ruff check src/ tests/ config/ demo/ scripts/

# Auto-fix lint issues
ruff check --fix src/ tests/ config/ demo/ scripts/

# Type-check
mypy src/ config/

# Format
ruff format src/ tests/ config/ demo/ scripts/
```

## Architecture

The system is a LangGraph state machine that processes a `RAGState` dict through a fixed pipeline:

```
load_documents → route_chunking → chunk_<strategy> → embed_chunks
    → store_vectors → retrieve → generate → END
```

Error paths at every stage route to `handle_error → END`. There is one dedicated graph node per chunking strategy; a conditional edge dispatches to the correct one based on `state["chunking_strategy"]`.

**Key invariants:**
- `config/settings.py` is the single source of truth for every constant — no magic numbers anywhere else.
- One responsibility per file: `retriever.py` only embeds + searches; `generator.py` only formats context + calls the LLM.
- `demo/` scripts are thin callers — all logic lives in `src/`.
- Every constant used system-wide is defined once in `config/settings.py` and imported everywhere.

## Layer Map

| Layer | Path | Responsibility |
|-------|------|---------------|
| Config | `config/settings.py` | All constants (model IDs, chunk sizes, top_k, timeouts) |
| Chunking | `src/chunking/` | 7 strategies + `ChunkingFactory` |
| RAG | `src/rag/` | S3 loader, Bedrock embeddings, OpenSearch vector store, retriever, generator |
| Graph | `src/graph/` | `RAGState` TypedDict, `build_*` node factories, `RAGWorkflow` |
| Agent | `src/agent/` | `BedrockAgentDeployer` — create/update/prepare/alias a Bedrock Agent |
| Infra | `terraform/` | All AWS resources as code |
| CI/CD | `.github/workflows/` | `test.yml` (lint + unit tests), `deploy.yml` (full deploy pipeline) |

## Chunking Strategies

`ChunkingFactory.create(strategy, **kwargs)` is the only valid instantiation path. Strategy names match `ChunkingStrategy` enum values: `fixed_size`, `recursive`, `markdown_aware`, `semantic`, `sentence`, `sliding_window`, `token_based`.

`semantic` is the only strategy that requires `embedding_fn=` at construction time (uses cosine similarity between sentence embeddings to detect topic boundaries). All other strategies are pure Python with no external dependencies.

## Local Testing

All unit tests in `tests/` run without live AWS credentials:

- `tests/conftest.py` — shared pytest fixtures (sample documents, chunks, mocked clients)
- `unittest.mock.patch("boto3.client")` — prevents real Bedrock/S3 calls
- `src/rag/in_memory_vector_store.py` — drop-in `OpenSearchVectorStore` replacement backed by numpy cosine similarity; used in tests that exercise the full store interface without OpenSearch

Tests marked `@pytest.mark.integration` require real AWS and are excluded from the default run with `-m "not integration"`.

## AWS Infrastructure

All resources live in `terraform/`. State is stored in an S3 backend configured at deploy time via `-backend-config` flags — see `.github/workflows/deploy.yml` for the exact flags.

GitHub Actions authenticates via OIDC (no long-lived credentials). The IAM role trusts only `refs/heads/main` pushes from the configured GitHub repository.
