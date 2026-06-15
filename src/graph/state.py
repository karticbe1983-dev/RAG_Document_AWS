"""LangGraph shared state TypedDict for the RAG pipeline.

Every node reads from and writes to this dict; LangGraph merges partial
updates returned by each node into the accumulated state.
"""

from typing import Any
from typing_extensions import TypedDict

from ..chunking.base import Chunk
from ..rag.document_loader import Document
from ..rag.vector_store import SearchResult


class RAGState(TypedDict):
    """Accumulated state passed between all nodes in the RAG workflow.

    Fields are grouped by pipeline stage:

    **Input** (set by ``RAGWorkflow.run()``)
        - question: Natural-language question to answer.
        - chunking_strategy: String value of the chosen ``ChunkingStrategy``.
        - top_k: Number of chunks to retrieve.
        - filters: Optional OpenSearch term filters.
        - s3_prefix: S3 key prefix used to filter loaded documents.
        - force_reindex: When True, chunks are re-embedded and re-stored.

    **Pipeline stages** (populated by individual nodes)
        - documents: Raw ``Document`` objects loaded from S3.
        - chunks: ``Chunk`` objects produced by the selected chunking strategy.
        - embeddings: Embedding vectors parallel to ``chunks``.
        - search_results: ``SearchResult`` objects returned by the retriever.

    **Output** (populated by the generate node)
        - answer: LLM-generated answer string.
        - sources: Serialised source citations (content preview + score + S3 URI).
        - error: Non-empty when any node encountered an unrecoverable error.
        - processing_steps: Human-readable log of what each node did.
        - token_usage: Input/output token counts from the LLM call.
    """

    # ── Input ─────────────────────────────────────────────────────────────────
    question: str
    chunking_strategy: str
    top_k: int
    filters: dict[str, Any]
    s3_prefix: str
    force_reindex: bool
    use_hybrid: bool

    # ── Pipeline stages ───────────────────────────────────────────────────────
    documents: list[Document]
    chunks: list[Chunk]
    embeddings: list[list[float]]
    search_results: list[SearchResult]

    # ── Output ────────────────────────────────────────────────────────────────
    answer: str
    sources: list[dict[str, Any]]
    error: str
    processing_steps: list[str]
    token_usage: dict[str, int]
