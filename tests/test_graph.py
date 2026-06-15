"""Unit tests for LangGraph workflow nodes and routing."""

import pytest
from unittest.mock import MagicMock, patch
from src.chunking.factory import ChunkingStrategy
from src.graph.state import RAGState
from src.graph.nodes import (
    build_chunk_documents_node,
    build_load_documents_node,
    check_for_errors,
    route_by_chunking_strategy,
)
from src.rag.document_loader import Document


def _make_state(**kwargs) -> RAGState:
    defaults: RAGState = {
        "question": "What is RAG?",
        "chunking_strategy": ChunkingStrategy.RECURSIVE.value,
        "top_k": 5,
        "filters": {},
        "s3_prefix": "",
        "force_reindex": False,
        "documents": [],
        "chunks": [],
        "embeddings": [],
        "search_results": [],
        "answer": "",
        "sources": [],
        "error": "",
        "processing_steps": [],
        "token_usage": {},
    }
    defaults.update(kwargs)
    return defaults


# ── Error routing ─────────────────────────────────────────────────────────────

class TestCheckForErrors:
    def test_continue_when_no_error(self):
        state = _make_state(error="")
        assert check_for_errors(state) == "continue"

    def test_error_when_error_set(self):
        state = _make_state(error="Something went wrong")
        assert check_for_errors(state) == "error"


# ── Chunking strategy routing ─────────────────────────────────────────────────

class TestRouteByChunkingStrategy:
    @pytest.mark.parametrize("strategy", [s.value for s in ChunkingStrategy])
    def test_valid_strategies(self, strategy: str):
        state = _make_state(chunking_strategy=strategy)
        assert route_by_chunking_strategy(state) == strategy

    def test_unknown_strategy_defaults_to_recursive(self):
        state = _make_state(chunking_strategy="not_a_real_strategy")
        result = route_by_chunking_strategy(state)
        assert result == ChunkingStrategy.RECURSIVE.value


# ── Document loading node ─────────────────────────────────────────────────────

class TestLoadDocumentsNode:
    def test_successful_load(self):
        mock_loader = MagicMock()
        mock_loader.load_all.return_value = [
            Document(content="# Doc 1\nContent", metadata={"source": "s3://b/d.md"}, document_id="d_md"),
        ]
        node = build_load_documents_node(mock_loader)
        state = _make_state(s3_prefix="docs/")
        result = node(state)
        assert len(result["documents"]) == 1
        assert "Loaded 1 documents" in result["processing_steps"][0]

    def test_loader_error_sets_error_state(self):
        mock_loader = MagicMock()
        mock_loader.load_all.side_effect = Exception("S3 not accessible")
        node = build_load_documents_node(mock_loader)
        result = node(_make_state())
        assert result["error"] == "S3 not accessible"
        assert result["documents"] == []


# ── Chunk documents node ───────────────────────────────────────────────────────

class TestChunkDocumentsNode:
    def _make_doc(self, content: str, doc_id: str = "test_doc") -> Document:
        return Document(
            content=content,
            metadata={"source": "s3://b/test.md", "document_id": doc_id},
            document_id=doc_id,
        )

    @pytest.mark.parametrize("strategy", [s.value for s in ChunkingStrategy if s != ChunkingStrategy.SEMANTIC])
    def test_all_non_semantic_strategies(self, strategy: str):
        text = "This is a test document. It has multiple sentences. " * 20
        doc = self._make_doc(text)
        state = _make_state(chunking_strategy=strategy, documents=[doc])
        node = build_chunk_documents_node()
        result = node(state)
        assert len(result["chunks"]) >= 1

    def test_accumulates_processing_steps(self):
        doc = self._make_doc("Some content " * 50)
        state = _make_state(
            chunking_strategy=ChunkingStrategy.FIXED_SIZE.value,
            documents=[doc],
            processing_steps=["Step 1 already done"],
        )
        node = build_chunk_documents_node()
        result = node(state)
        assert len(result["processing_steps"]) == 2
        assert "Step 1 already done" in result["processing_steps"][0]

    def test_empty_documents_returns_empty_chunks(self):
        state = _make_state(documents=[])
        node = build_chunk_documents_node()
        result = node(state)
        assert result["chunks"] == []
