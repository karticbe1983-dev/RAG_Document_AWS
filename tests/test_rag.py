"""Unit tests for RAG core modules (mocked AWS calls)."""

import json
import pytest
from unittest.mock import MagicMock, patch, call
from src.rag.document_loader import S3DocumentLoader, Document
from src.rag.embeddings import BedrockEmbeddings
from src.rag.retriever import RAGRetriever


# ── Document Loader ───────────────────────────────────────────────────────────

class TestS3DocumentLoader:
    def _make_loader(self, bucket="test-bucket"):
        with patch("boto3.client"):
            loader = S3DocumentLoader(bucket_name=bucket, region="us-east-1")
        return loader

    def test_load_document_success(self):
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

    def test_load_document_client_error(self):
        from botocore.exceptions import ClientError
        loader = self._make_loader()
        loader._s3.get_object.side_effect = ClientError(
            {"Error": {"Code": "NoSuchKey", "Message": "Not found"}}, "GetObject"
        )
        doc = loader.load_document("missing.md")
        assert doc is None

    def test_load_all_filters_by_extension(self):
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
    def _make_embeddings(self, model_id="amazon.titan-embed-text-v2:0"):
        with patch("boto3.client"):
            emb = BedrockEmbeddings(model_id=model_id, region="us-east-1")
        return emb

    def test_embed_returns_vector(self):
        emb = self._make_embeddings()
        expected_vector = [0.1] * 1024
        mock_response = MagicMock()
        mock_response.__getitem__.side_effect = lambda k: (
            MagicMock(read=lambda: json.dumps({"embedding": expected_vector}).encode())
            if k == "body" else None
        )
        emb._client.invoke_model.return_value = {
            "body": MagicMock(read=lambda: json.dumps({"embedding": expected_vector}).encode())
        }
        result = emb.embed("test text")
        assert len(result) == 1024
        assert result == expected_vector

    def test_embed_batch_calls_embed_per_text(self):
        emb = self._make_embeddings()
        emb.embed = MagicMock(return_value=[0.1] * 1024)
        texts = ["text one", "text two", "text three"]
        results = emb.embed_batch(texts)
        assert len(results) == 3
        assert emb.embed.call_count == 3

    def test_request_body_v2_format(self):
        emb = self._make_embeddings("amazon.titan-embed-text-v2:0")
        body = emb._build_request_body("hello")
        assert "dimensions" in body
        assert body["normalize"] is True

    def test_request_body_v1_format(self):
        emb = self._make_embeddings("amazon.titan-embed-text-v1")
        body = emb._build_request_body("hello")
        assert "dimensions" not in body
        assert body["inputText"] == "hello"


# ── RAG Retriever ─────────────────────────────────────────────────────────────

class TestRAGRetriever:
    def _make_retriever(self):
        with patch("boto3.client"):
            embeddings = MagicMock()
            embeddings.embed.return_value = [0.1] * 1024
            vector_store = MagicMock()
            retriever = RAGRetriever(
                embeddings=embeddings,
                vector_store=vector_store,
                llm_model_id="anthropic.claude-3-5-sonnet-20241022-v2:0",
                region="us-east-1",
            )
        return retriever, embeddings, vector_store

    def test_query_calls_embed_and_search(self):
        retriever, embeddings, vector_store = self._make_retriever()
        mock_chunk = MagicMock()
        mock_chunk.content = "Relevant content here."
        mock_chunk.metadata = {"source": "s3://bucket/doc.md"}
        mock_chunk.chunk_id = "chunk_1"
        vector_store.similarity_search.return_value = [MagicMock(chunk=mock_chunk, score=0.9)]

        answer_body = {
            "content": [{"text": "This is the answer."}],
            "usage": {"input_tokens": 100, "output_tokens": 50},
        }
        retriever._bedrock.invoke_model.return_value = {
            "body": MagicMock(read=lambda: json.dumps(answer_body).encode())
        }

        response = retriever.query("What is RAG?")
        embeddings.embed.assert_called_once_with("What is RAG?")
        vector_store.similarity_search.assert_called_once()
        assert response.answer == "This is the answer."
        assert response.input_tokens == 100
        assert response.output_tokens == 50

    def test_format_context_includes_source(self):
        retriever, _, _ = self._make_retriever()
        mock_result = MagicMock()
        mock_result.chunk.content = "Some content"
        mock_result.chunk.metadata = {"source": "s3://bucket/doc.md"}
        context = retriever._format_context([mock_result])
        assert "s3://bucket/doc.md" in context
        assert "Some content" in context
