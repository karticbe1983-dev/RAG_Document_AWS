"""Vector-search retriever: embeds a question and returns the top-k matching chunks."""

import logging
from typing import Any

from config.settings import DEFAULT_TOP_K
from .embeddings import BedrockEmbeddings
from .vector_store import OpenSearchVectorStore, SearchResult

logger = logging.getLogger(__name__)


class RAGRetriever:
    """Retrieve semantically relevant chunks for a question using k-NN vector search.

    Responsibility: embed the query with Bedrock Titan, then call the
    OpenSearch k-NN index and return ranked SearchResult objects.
    Generation is handled separately by RAGGenerator.
    """

    def __init__(
        self,
        embeddings: BedrockEmbeddings,
        vector_store: OpenSearchVectorStore,
    ) -> None:
        """Initialise the retriever with pre-built embedding and store clients.

        Args:
            embeddings: Bedrock embeddings client used to embed the question.
            vector_store: OpenSearch vector store to search against.
        """
        self.embeddings = embeddings
        self.vector_store = vector_store

    def retrieve(
        self,
        question: str,
        top_k: int = DEFAULT_TOP_K,
        filters: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        """Embed *question* and return the *top_k* most similar chunks.

        Args:
            question: Natural-language question to search for.
            top_k: Number of chunks to return.
            filters: Optional OpenSearch term filters (e.g. ``{"document_id": "doc_1"}``).

        Returns:
            List of SearchResult objects ordered by descending similarity score.
        """
        query_embedding = self.embeddings.embed(question)
        return self.vector_store.similarity_search(
            query_embedding, top_k=top_k, filters=filters
        )
