"""FastAPI server exposing the RAG pipeline as an HTTP API.

Run locally (no AWS needed)::

    LOCAL_MODE=true uvicorn src.api.app:app --reload --port 8000

Run against real AWS::

    S3_BUCKET_NAME=my-bucket OPENSEARCH_ENDPOINT=abc.us-east-1.aoss.amazonaws.com \\
        uvicorn src.api.app:app --reload --port 8000

Or directly::

    python -m src.api.app

Environment variables:
    LOCAL_MODE              — "true" to run without AWS (uses local docs/ folder)
    S3_BUCKET_NAME          — S3 bucket holding the knowledge documents (AWS mode)
    OPENSEARCH_ENDPOINT     — OpenSearch Serverless collection hostname (AWS mode)
    AWS_REGION              — defaults to us-east-1
    OPENSEARCH_INDEX_NAME   — defaults to rag-index
"""

import logging
import os
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Any

from config.settings import AWS_REGION, DEFAULT_TOP_K, OPENSEARCH_INDEX_NAME
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.chunking.factory import ChunkingFactory, ChunkingStrategy
from src.graph.workflow import RAGWorkflow, WorkflowConfig

logger = logging.getLogger(__name__)


# ── Request / response models ─────────────────────────────────────────────────

class QueryRequest(BaseModel):
    """Body for POST /query."""

    question: str = Field(..., description="Natural-language question to answer")
    chunking_strategy: str = Field(
        ChunkingStrategy.RECURSIVE.value,
        description="One of: fixed_size, recursive, markdown, semantic, sentence, sliding_window, token",
    )
    top_k: int = Field(DEFAULT_TOP_K, ge=1, le=20, description="Number of chunks to retrieve")
    s3_prefix: str = Field("", description="S3 key prefix to filter documents (e.g. 'docs/')")
    force_reindex: bool = Field(False, description="Re-embed and re-index all documents before querying")
    filters: dict[str, Any] | None = Field(None, description="Optional field-equality filters")


class SourceCitation(BaseModel):
    """A single retrieved chunk included in the answer."""

    content: str
    score: float
    source: str
    chunk_id: str


class QueryResponse(BaseModel):
    """Result returned by POST /query."""

    question: str
    answer: str
    sources: list[SourceCitation]
    chunking_strategy: str
    processing_steps: list[str]
    token_usage: dict[str, int]
    error: str


class StrategiesResponse(BaseModel):
    """List of available chunking strategy names."""

    strategies: list[str]


class HealthResponse(BaseModel):
    """Service liveness and configuration status."""

    status: str
    workflow_ready: bool


# ── App lifespan ──────────────────────────────────────────────────────────────

def _build_local_workflow() -> RAGWorkflow:
    """Assemble a fully local workflow and pre-index all docs at startup.

    Uses the local docs/ directory, hash-based embeddings, an in-memory
    vector store, and a context-passthrough generator.  Documents are chunked
    and indexed once at startup so every query works without force_reindex=true.
    """
    from src.chunking.factory import ChunkingFactory, ChunkingStrategy
    from src.rag.in_memory_vector_store import InMemoryVectorStore
    from src.rag.local_document_loader import LocalDocumentLoader
    from src.rag.local_embeddings import LocalEmbeddings
    from src.rag.local_generator import LocalGenerator

    local_loader = LocalDocumentLoader("docs")
    local_embeddings = LocalEmbeddings()
    store = InMemoryVectorStore()

    # Index all docs once so queries work immediately (no force_reindex needed)
    docs = local_loader.load_all()
    if docs:
        chunker = ChunkingFactory.create(ChunkingStrategy.RECURSIVE)
        chunks = []
        for doc in docs:
            chunks.extend(chunker.chunk(doc.content, metadata={**doc.metadata, "document_id": doc.document_id}))
        embeddings_list = local_embeddings.embed_batch([c.content for c in chunks])
        store.create_index()
        store.add_chunks(chunks, embeddings_list)
        logger.info("Pre-indexed %d chunks from %d local documents", len(chunks), len(docs))

    config = WorkflowConfig(s3_bucket="local", opensearch_endpoint="local")
    return RAGWorkflow(
        config,
        loader=local_loader,
        embeddings=local_embeddings,
        vector_store=store,
        generator=LocalGenerator(),
    )


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Initialise the RAGWorkflow once at startup using environment variables."""
    local_mode = os.getenv("LOCAL_MODE", "false").lower() == "true"
    s3_bucket = os.getenv("S3_BUCKET_NAME", "")
    opensearch_endpoint = os.getenv("OPENSEARCH_ENDPOINT", "")

    if local_mode:
        try:
            app.state.workflow = _build_local_workflow()
            logger.info("RAGWorkflow ready in LOCAL MODE (docs/ folder, no AWS)")
        except Exception as exc:
            logger.error("Local workflow init failed: %s", exc)
            app.state.workflow = None
    elif s3_bucket and opensearch_endpoint:
        try:
            config = WorkflowConfig(
                s3_bucket=s3_bucket,
                opensearch_endpoint=opensearch_endpoint,
                aws_region=os.getenv("AWS_REGION", AWS_REGION),
                opensearch_index=os.getenv("OPENSEARCH_INDEX_NAME", OPENSEARCH_INDEX_NAME),
            )
            app.state.workflow = RAGWorkflow(config)
            logger.info("RAGWorkflow ready (bucket=%s)", s3_bucket)
        except Exception as exc:
            logger.error("RAGWorkflow init failed: %s", exc)
            app.state.workflow = None
    else:
        app.state.workflow = None
        logger.warning(
            "No workflow configured. Set LOCAL_MODE=true for local testing, "
            "or set S3_BUCKET_NAME + OPENSEARCH_ENDPOINT for AWS mode."
        )

    yield


# ── App instance ──────────────────────────────────────────────────────────────

app = FastAPI(
    title="RAG Document API",
    description=(
        "LangGraph RAG pipeline with 7 chunking strategies "
        "backed by AWS Bedrock (Claude + Titan) and OpenSearch Serverless."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["ops"])
def health() -> HealthResponse:
    """Return service liveness and whether the workflow is ready to accept queries."""
    return HealthResponse(
        status="ok",
        workflow_ready=app.state.workflow is not None,
    )


@app.get("/strategies", response_model=StrategiesResponse, tags=["chunking"])
def list_strategies() -> StrategiesResponse:
    """List every available chunking strategy name."""
    return StrategiesResponse(strategies=ChunkingFactory.available_strategies())


@app.post("/query", response_model=QueryResponse, tags=["rag"])
def query(request: QueryRequest) -> QueryResponse:
    """Run the full RAG pipeline and return a grounded answer with source citations.

    The pipeline: load S3 docs → chunk → embed → store → retrieve → generate.
    """
    if app.state.workflow is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "Workflow not ready. "
                "Set S3_BUCKET_NAME and OPENSEARCH_ENDPOINT environment variables and restart."
            ),
        )

    result = app.state.workflow.run(
        question=request.question,
        chunking_strategy=request.chunking_strategy,
        top_k=request.top_k,
        s3_prefix=request.s3_prefix,
        force_reindex=request.force_reindex,
        filters=request.filters,
    )

    return QueryResponse(
        question=result["question"],
        answer=result["answer"],
        sources=[SourceCitation(**s) for s in result["sources"]],
        chunking_strategy=result["chunking_strategy"],
        processing_steps=result["processing_steps"],
        token_usage=result["token_usage"],
        error=result["error"],
    )


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.api.app:app", host="0.0.0.0", port=8000, reload=True)
