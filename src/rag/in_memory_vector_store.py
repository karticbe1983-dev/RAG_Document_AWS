"""In-memory vector store backed by numpy cosine similarity.

Drop-in replacement for OpenSearchVectorStore for local runs and unit tests
that need to exercise the full store interface without an OpenSearch endpoint.
"""

from typing import Any

import numpy as np

from ..chunking.base import Chunk
from .vector_store import SearchResult


class InMemoryVectorStore:
    """Numpy-backed vector store implementing the OpenSearchVectorStore interface.

    Responsibility: store chunks with their embedding vectors in memory and
    answer k-NN queries via cosine similarity.  No network calls are made.
    """

    def __init__(self) -> None:
        """Initialise an empty in-memory store."""
        self._chunks: list[Chunk] = []
        self._embeddings: list[list[float]] = []

    def create_index(self, dimension: int | None = None) -> bool:
        """No-op: an in-memory store needs no index initialisation.

        Args:
            dimension: Ignored.

        Returns:
            Always ``True``.
        """
        return True

    def add_chunks(self, chunks: list[Chunk], embeddings: list[list[float]]) -> int:
        """Append chunks and their embeddings to the in-memory store.

        Args:
            chunks: Chunk objects to store.
            embeddings: Parallel list of embedding vectors; length must match *chunks*.

        Returns:
            Number of chunks added.

        Raises:
            ValueError: If *chunks* and *embeddings* have different lengths.
        """
        if len(chunks) != len(embeddings):
            raise ValueError("Chunks and embeddings must have the same length")
        self._chunks.extend(chunks)
        self._embeddings.extend(embeddings)
        return len(chunks)

    def similarity_search(
        self,
        query_embedding: list[float],
        top_k: int,
        filters: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        """Return the *top_k* chunks most similar to *query_embedding*.

        Similarity is cosine similarity computed with numpy.  Optional *filters*
        are applied as exact-match equality checks on chunk metadata fields.

        Args:
            query_embedding: Dense query vector.
            top_k: Maximum number of results to return.
            filters: Dict of ``{metadata_field: value}`` equality filters.

        Returns:
            SearchResult list ordered by descending cosine similarity score.
        """
        if not self._embeddings:
            return []

        q = np.array(query_embedding, dtype=float)
        q_norm = float(np.linalg.norm(q))
        if q_norm == 0.0:
            return []

        scored: list[tuple[float, int]] = []
        for i, emb in enumerate(self._embeddings):
            if filters and not self._matches_filters(self._chunks[i], filters):
                continue
            e = np.array(emb, dtype=float)
            e_norm = float(np.linalg.norm(e))
            score = float(np.dot(q, e) / (q_norm * e_norm)) if e_norm != 0.0 else 0.0
            scored.append((score, i))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [
            SearchResult(chunk=self._chunks[idx], score=score)
            for score, idx in scored[:top_k]
        ]

    def _matches_filters(self, chunk: Chunk, filters: dict[str, Any]) -> bool:
        """Return True only if *chunk* satisfies all equality filters.

        Args:
            chunk: Chunk to evaluate.
            filters: Dict of ``{metadata_field: expected_value}``.

        Returns:
            ``True`` when every filter field matches; ``False`` otherwise.
        """
        return all(chunk.metadata.get(k) == v for k, v in filters.items())
