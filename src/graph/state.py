from typing import Any
from typing_extensions import TypedDict
from ..chunking.base import Chunk
from ..rag.document_loader import Document
from ..rag.vector_store import SearchResult


class RAGState(TypedDict):
    # Input
    question: str
    chunking_strategy: str
    top_k: int
    filters: dict[str, Any]
    s3_prefix: str
    force_reindex: bool

    # Pipeline stages
    documents: list[Document]
    chunks: list[Chunk]
    embeddings: list[list[float]]
    search_results: list[SearchResult]

    # Output
    answer: str
    sources: list[dict[str, Any]]
    error: str
    processing_steps: list[str]
    token_usage: dict[str, int]
