"""Unit tests for RAG core modules (all AWS calls are mocked)."""

import json
import pytest
from unittest.mock import MagicMock, patch

from src.rag.document_loader import S3DocumentLoader, Document
from src.rag.embeddings import BedrockEmbeddings
from src.rag.retriever import RAGRetriever
from src.rag.generator import RAGGenerator, RAGResponse
from src.rag.vector_store import SearchResult
from src.chunking.base import Chunk


# ── Document Loader ───────────────────────────────────────────────────────────

class TestS3DocumentLoader:
    def _make_loader(self, bucket: str = "test-bucket") -> S3DocumentLoader:
        with patch("boto3.client"):
            loader = S3DocumentLoader(bucket_name=bucket, region="us-east-1")
        return loader

    def test_load_document_returns_document_on_success(self) -> None:
        """load_document decodes body bytes and populates all metadata fields."""
        loader = self._make_loader()
        mock_body = MagicMock()
        mock_body.read.return_value = b"# Hello\nThis is content."
        loader._s3.get_object.return_value = {
            "Body": mock_body,
            "ContentType": "text/markdown",
            "LastModified": "2024-01-01",
            "ContentLength": 24,
        }
        doc = loader.load_document("docs/hello.md")
        assert doc is not None
        assert doc.content == "# Hello\nThis is content."
        assert doc.metadata["source"] == "s3://test-bucket/docs/hello.md"
        assert doc.document_id == "docs_hello_md"

    def test_load_document_returns_none_on_client_error(self) -> None:
        """load_document returns None and logs the error instead of raising."""
        from botocore.exceptions import ClientError
        loader = self._make_loader()
        loader._s3.get_object.side_effect = ClientError(
            {"Error": {"Code": "NoSuchKey", "Message": "Not found"}}, "GetObject"
        )
        assert loader.load_document("missing.md") is None

    def test_list_objects_filters_by_extension(self) -> None:
        """_list_objects excludes keys whose suffix is not in the allowed list."""
        loader = self._make_loader()
        loader._s3.get_paginator.return_value.paginate.return_value = [
            {
                "Contents": [
                    {"Key": "docs/file.md"},
                    {"Key": "docs/file.pdf"},
                    {"Key": "docs/file.txt"},
                ]
            }
        ]
        keys = loader._list_objects("docs/", [".md", ".txt"])
        assert "docs/file.md" in keys
        assert "docs/file.txt" in keys
        assert "docs/file.pdf" not in keys


# ── Bedrock Embeddings ────────────────────────────────────────────────────────

class TestBedrockEmbeddings:
    def _make_embeddings(
        self, model_id: str = "amazon.titan-embed-text-v2:0"
    ) -> BedrockEmbeddings:
        with patch("boto3.client"):
            return BedrockEmbeddings(model_id=model_id, region="us-east-1")

    def test_embed_returns_vector_of_correct_length(self) -> None:
        """embed() calls InvokeModel and returns the embedding from the response."""
        emb = self._make_embeddings()
        expected_vector = [0.1] * 1024
        emb._client.invoke_model.return_value = {
            "body": MagicMock(read=lambda: json.dumps({"embedding": expected_vector}).encode())
        }
        result = emb.embed("test text")
        assert result == expected_vector

    def test_embed_batch_calls_embed_for_each_text(self) -> None:
        """embed_batch() delegates to embed() once per input string."""
        emb = self._make_embeddings()
        emb.embed = MagicMock(return_value=[0.1] * 1024)
        results = emb.embed_batch(["a", "b", "c"])
        assert len(results) == 3
        assert emb.embed.call_count == 3

    def test_call_delegates_to_embed(self) -> None:
        """__call__ is an alias for embed (used as embedding_fn in SemanticChunker)."""
        emb = self._make_embeddings()
        emb.embed = MagicMock(return_value=[0.5] * 1024)
        result = emb("hello world")
        emb.embed.assert_called_once_with("hello world")
        assert result == [0.5] * 1024

    def test_request_body_v2_includes_dimensions_and_normalize(self) -> None:
        """Titan v2 body must include 'dimensions' and 'normalize'."""
        emb = self._make_embeddings("amazon.titan-embed-text-v2:0")
        body = emb._build_request_body("hello")
        assert "dimensions" in body
        assert body["normalize"] is True

    def test_request_body_v1_omits_dimensions(self) -> None:
        """Titan v1 body must NOT include 'dimensions'."""
        emb = self._make_embeddings("amazon.titan-embed-text-v1")
        body = emb._build_request_body("hello")
        assert "dimensions" not in body
        assert body["inputText"] == "hello"


# ── RAG Retriever ─────────────────────────────────────────────────────────────

class TestRAGRetriever:
    def _make_retriever(self) -> tuple[RAGRetriever, MagicMock, MagicMock]:
        embeddings = MagicMock()
        embeddings.embed.return_value = [0.1] * 1024
        vector_store = MagicMock()
        retriever = RAGRetriever(embeddings=embeddings, vector_store=vector_store)
        return retriever, embeddings, vector_store

    def test_retrieve_embeds_question_then_searches(self) -> None:
        """retrieve() calls embed() once and passes the result to similarity_search()."""
        retriever, embeddings, vector_store = self._make_retriever()
        vector_store.similarity_search.return_value = []
        retriever.retrieve("What is RAG?", top_k=3)
        embeddings.embed.assert_called_once_with("What is RAG?")
        vector_store.similarity_search.assert_called_once_with(
            [0.1] * 1024, top_k=3, filters=None
        )

    def test_retrieve_forwards_filters(self) -> None:
        """retrieve() passes the filters dict through to similarity_search()."""
        retriever, _, vector_store = self._make_retriever()
        vector_store.similarity_search.return_value = []
        retriever.retrieve("q", top_k=5, filters={"document_id": "doc_1"})
        _, kwargs = vector_store.similarity_search.call_args
        assert kwargs["filters"] == {"document_id": "doc_1"}

    def test_retrieve_returns_search_results(self) -> None:
        """retrieve() returns whatever similarity_search() returns."""
        retriever, _, vector_store = self._make_retriever()
        fake_result = MagicMock(spec=SearchResult)
        vector_store.similarity_search.return_value = [fake_result]
        results = retriever.retrieve("q")
        assert results == [fake_result]


# ── RAG Generator ─────────────────────────────────────────────────────────────

class TestRAGGenerator:
    def _make_generator(self) -> RAGGenerator:
        with patch("boto3.client"):
            return RAGGenerator(region="us-east-1")

    def _make_source(self, content: str = "Some content") -> SearchResult:
        chunk = Chunk(
            content=content,
            metadata={"source": "s3://bucket/doc.md"},
            chunk_id="chunk_1",
            document_id="doc",
        )
        return SearchResult(chunk=chunk, score=0.9)

    def test_generate_returns_rag_response(self) -> None:
        """generate() returns a RAGResponse with answer text and source list."""
        gen = self._make_generator()
        answer_body = {
            "content": [{"text": "This is the answer."}],
            "usage": {"input_tokens": 100, "output_tokens": 50},
        }
        gen._bedrock.invoke_model.return_value = {
            "body": MagicMock(read=lambda: json.dumps(answer_body).encode())
        }
        response = gen.generate("What is RAG?", [self._make_source()])
        assert isinstance(response, RAGResponse)
        assert response.answer == "This is the answer."
        assert response.input_tokens == 100
        assert response.output_tokens == 50

    def test_format_context_includes_source_and_content(self) -> None:
        """_format_context() tags each result with its source URL."""
        gen = self._make_generator()
        source = self._make_source("Relevant chunk content")
        context = gen._format_context([source])
        assert "s3://bucket/doc.md" in context
        assert "Relevant chunk content" in context

    def test_format_context_numbers_each_source(self) -> None:
        """_format_context() prefixes each result with [1], [2], etc."""
        gen = self._make_generator()
        sources = [self._make_source("A"), self._make_source("B")]
        context = gen._format_context(sources)
        assert "[1]" in context
        assert "[2]" in context
