"""Unit tests for LangGraph workflow nodes and routing logic."""

from unittest.mock import MagicMock

import pytest
from src.chunking.base import Chunk
from src.chunking.factory import ChunkingStrategy
from src.graph.nodes import (
    build_chunk_documents_node,
    build_generate_node,
    build_load_documents_node,
    build_retrieve_node,
    check_for_errors,
)
from src.graph.state import RAGState
from src.rag.document_loader import Document
from src.rag.vector_store import SearchResult


def _make_state(**overrides) -> RAGState:
    """Return a fully populated RAGState with sensible defaults."""
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
    defaults.update(overrides)
    return defaults


def _make_doc(content: str, doc_id: str = "test_doc") -> Document:
    """Return a minimal Document for use in node tests."""
    return Document(
        content=content,
        metadata={"source": f"s3://b/{doc_id}.md", "document_id": doc_id},
        document_id=doc_id,
    )


# ── check_for_errors ──────────────────────────────────────────────────────────

class TestCheckForErrors:
    def test_returns_continue_when_no_error(self) -> None:
        """Empty error string routes to 'continue'."""
        assert check_for_errors(_make_state(error="")) == "continue"

    def test_returns_error_when_error_set(self) -> None:
        """Non-empty error string routes to 'error'."""
        assert check_for_errors(_make_state(error="boom")) == "error"

    def test_returns_continue_for_missing_error_key(self) -> None:
        """Missing 'error' key is treated as no error."""
        state = _make_state()
        state.pop("error")  # type: ignore[misc]
        assert check_for_errors(state) == "continue"


# ── build_load_documents_node ─────────────────────────────────────────────────

class TestLoadDocumentsNode:
    def test_stores_documents_and_records_step(self) -> None:
        """Successful load stores docs in state and appends a processing step."""
        mock_loader = MagicMock()
        mock_loader.load_all.return_value = [_make_doc("# Doc 1\nContent")]
        node = build_load_documents_node(mock_loader)
        result = node(_make_state(s3_prefix="docs/"))
        assert len(result["documents"]) == 1
        assert "Loaded 1 documents" in result["processing_steps"][0]

    def test_sets_error_state_on_exception(self) -> None:
        """Exception from load_all is caught and stored in 'error' key."""
        mock_loader = MagicMock()
        mock_loader.load_all.side_effect = Exception("S3 not accessible")
        node = build_load_documents_node(mock_loader)
        result = node(_make_state())
        assert result["error"] == "S3 not accessible"
        assert result["documents"] == []

    def test_passes_s3_prefix_from_state(self) -> None:
        """The node forwards 'state.s3_prefix' to loader.load_all()."""
        mock_loader = MagicMock()
        mock_loader.load_all.return_value = []
        node = build_load_documents_node(mock_loader)
        node(_make_state(s3_prefix="archive/"))
        mock_loader.load_all.assert_called_once_with(prefix="archive/")


# ── build_chunk_documents_node ────────────────────────────────────────────────

class TestChunkDocumentsNode:
    @pytest.mark.parametrize(
        "strategy",
        [s.value for s in ChunkingStrategy if s != ChunkingStrategy.SEMANTIC],
    )
    def test_produces_chunks_for_all_non_semantic_strategies(self, strategy: str) -> None:
        """Every non-semantic strategy returns at least one chunk for non-trivial text."""
        text = "This is a test sentence. " * 30
        state = _make_state(chunking_strategy=strategy, documents=[_make_doc(text)])
        node = build_chunk_documents_node()
        result = node(state)
        assert len(result["chunks"]) >= 1

    def test_appends_step_to_existing_steps(self) -> None:
        """Processing steps from earlier nodes are preserved and extended."""
        doc = _make_doc("Some content " * 50)
        state = _make_state(
            chunking_strategy=ChunkingStrategy.FIXED_SIZE.value,
            documents=[doc],
            processing_steps=["Step 1 already done"],
        )
        result = build_chunk_documents_node()(state)
        assert len(result["processing_steps"]) == 2
        assert result["processing_steps"][0] == "Step 1 already done"

    def test_returns_empty_chunks_for_empty_document_list(self) -> None:
        """No documents → no chunks."""
        result = build_chunk_documents_node()(_make_state(documents=[]))
        assert result["chunks"] == []


# ── build_retrieve_node ───────────────────────────────────────────────────────

class TestRetrieveNode:
    def _make_result(self) -> SearchResult:
        chunk = Chunk(content="ctx", metadata={"source": "s3://b/d.md"}, chunk_id="c1")
        return SearchResult(chunk=chunk, score=0.9)

    def test_calls_retriever_with_question_and_top_k(self) -> None:
        """The node forwards question and top_k from state to retriever.retrieve()."""
        mock_retriever = MagicMock()
        mock_retriever.retrieve.return_value = [self._make_result()]
        node = build_retrieve_node(mock_retriever)
        node(_make_state(question="What is RAG?", top_k=3))
        mock_retriever.retrieve.assert_called_once_with("What is RAG?", top_k=3, filters=None)

    def test_stores_results_in_search_results(self) -> None:
        """Retrieved results are stored in 'search_results' key."""
        mock_retriever = MagicMock()
        results = [self._make_result()]
        mock_retriever.retrieve.return_value = results
        node = build_retrieve_node(mock_retriever)
        state_update = node(_make_state())
        assert state_update["search_results"] == results

    def test_sets_error_on_retrieval_failure(self) -> None:
        """Exception from retriever is caught and stored in 'error'."""
        mock_retriever = MagicMock()
        mock_retriever.retrieve.side_effect = Exception("OpenSearch unreachable")
        node = build_retrieve_node(mock_retriever)
        result = node(_make_state())
        assert result["error"] == "OpenSearch unreachable"
        assert result["search_results"] == []


# ── build_generate_node ───────────────────────────────────────────────────────

class TestGenerateNode:
    def _make_result(self) -> SearchResult:
        chunk = Chunk(content="context text", metadata={"source": "s3://b/d.md"}, chunk_id="c1")
        return SearchResult(chunk=chunk, score=0.85)

    def test_passes_search_results_from_state_to_generator(self) -> None:
        """The node uses search_results already in state (no re-retrieval)."""
        from src.rag.generator import RAGResponse
        mock_generator = MagicMock()
        results = [self._make_result()]
        mock_generator.generate.return_value = RAGResponse(
            answer="The answer.",
            sources=results,
            model_id="test-model",
            input_tokens=10,
            output_tokens=5,
        )
        node = build_generate_node(mock_generator)
        state = _make_state(question="Test?", search_results=results)
        node(state)
        mock_generator.generate.assert_called_once_with("Test?", results)

    def test_answer_and_token_usage_stored_in_result(self) -> None:
        """generate node populates 'answer' and 'token_usage' in its return dict."""
        from src.rag.generator import RAGResponse
        mock_generator = MagicMock()
        results = [self._make_result()]
        mock_generator.generate.return_value = RAGResponse(
            answer="42 is the answer.",
            sources=results,
            model_id="test-model",
            input_tokens=200,
            output_tokens=30,
        )
        node = build_generate_node(mock_generator)
        result = node(_make_state(search_results=results))
        assert result["answer"] == "42 is the answer."
        assert result["token_usage"]["input_tokens"] == 200
        assert result["token_usage"]["output_tokens"] == 30

    def test_sets_error_on_generation_failure(self) -> None:
        """Exception from generator is caught and stored in 'error'."""
        mock_generator = MagicMock()
        mock_generator.generate.side_effect = Exception("Bedrock throttled")
        node = build_generate_node(mock_generator)
        result = node(_make_state())
        assert result["error"] == "Bedrock throttled"
        assert result["answer"] == ""
