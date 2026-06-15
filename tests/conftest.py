"""Shared pytest fixtures for all test modules.

All fixtures here avoid real AWS calls.  AWS clients are replaced with
MagicMock instances so tests run without credentials or deployed infrastructure.
"""

import json
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from src.chunking.base import Chunk
from src.rag.document_loader import Document, S3DocumentLoader
from src.rag.embeddings import BedrockEmbeddings
from src.rag.generator import RAGGenerator
from src.rag.in_memory_vector_store import InMemoryVectorStore
from src.rag.retriever import RAGRetriever
from src.rag.vector_store import SearchResult


# ── Primitive data objects ────────────────────────────────────────────────────

@pytest.fixture()
def sample_chunk() -> Chunk:
    """A minimal Chunk with enough metadata to pass through the pipeline."""
    return Chunk(
        content="Retrieval-Augmented Generation (RAG) combines retrieval with generation.",
        metadata={"source": "s3://test-bucket/docs/intro.md", "document_id": "docs_intro_md"},
        chunk_id="chunk-001",
        document_id="docs_intro_md",
    )


@pytest.fixture()
def sample_document() -> Document:
    """A minimal Document as returned by S3DocumentLoader."""
    return Document(
        content="# Introduction\n\nRAG is a technique that augments LLM generation with retrieved documents.",
        metadata={
            "source": "s3://test-bucket/docs/intro.md",
            "key": "docs/intro.md",
            "bucket": "test-bucket",
            "document_id": "docs_intro_md",
        },
        document_id="docs_intro_md",
    )


@pytest.fixture()
def sample_search_result(sample_chunk: Chunk) -> SearchResult:
    """A SearchResult wrapping sample_chunk with a high similarity score."""
    return SearchResult(chunk=sample_chunk, score=0.92)


@pytest.fixture()
def unit_embedding() -> list[float]:
    """A 1024-dimensional unit vector used as a stand-in for a real embedding."""
    vec = [0.0] * 1024
    vec[0] = 1.0
    return vec


# ── Mocked AWS clients ────────────────────────────────────────────────────────

@pytest.fixture()
def mock_s3_loader() -> S3DocumentLoader:
    """S3DocumentLoader whose boto3 client is replaced with a MagicMock."""
    with patch("boto3.client"):
        loader = S3DocumentLoader(bucket_name="test-bucket", region="us-east-1")
    return loader


@pytest.fixture()
def mock_embeddings(unit_embedding: list[float]) -> BedrockEmbeddings:
    """BedrockEmbeddings whose boto3 client is a MagicMock.

    ``embed()`` and ``embed_batch()`` return unit vectors so callers get valid
    float lists without hitting Bedrock.
    """
    with patch("boto3.client"):
        emb = BedrockEmbeddings(model_id="amazon.titan-embed-text-v2:0", region="us-east-1")

    response_body = json.dumps({"embedding": unit_embedding}).encode()
    emb._client.invoke_model.return_value = {
        "body": MagicMock(read=lambda: response_body)
    }
    return emb


@pytest.fixture()
def mock_generator() -> RAGGenerator:
    """RAGGenerator whose boto3 client is a MagicMock.

    ``invoke_model`` returns a fixed answer so tests can assert on the response
    structure without calling Bedrock.
    """
    with patch("boto3.client"):
        gen = RAGGenerator(region="us-east-1")

    answer_body = json.dumps({
        "content": [{"text": "This is a test answer."}],
        "usage": {"input_tokens": 50, "output_tokens": 20},
    }).encode()
    gen._bedrock.invoke_model.return_value = {
        "body": MagicMock(read=lambda: answer_body)
    }
    return gen


# ── In-memory vector store ────────────────────────────────────────────────────

@pytest.fixture()
def memory_store() -> InMemoryVectorStore:
    """Empty InMemoryVectorStore, ready to accept chunks."""
    return InMemoryVectorStore()


@pytest.fixture()
def populated_store(
    sample_chunk: Chunk,
    unit_embedding: list[float],
) -> InMemoryVectorStore:
    """InMemoryVectorStore pre-populated with one chunk."""
    store = InMemoryVectorStore()
    store.add_chunks([sample_chunk], [unit_embedding])
    return store


# ── Assembled pipeline components ─────────────────────────────────────────────

@pytest.fixture()
def mock_retriever(
    mock_embeddings: BedrockEmbeddings,
    memory_store: InMemoryVectorStore,
) -> RAGRetriever:
    """RAGRetriever wired to mock embeddings and an in-memory vector store."""
    return RAGRetriever(embeddings=mock_embeddings, vector_store=memory_store)  # type: ignore[arg-type]
