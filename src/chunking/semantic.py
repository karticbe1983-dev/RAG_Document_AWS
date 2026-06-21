"""Semantic chunker that groups sentences by embedding-space similarity."""

import re
from typing import Any

import numpy as np
from config.settings import (
    SEARCH_PREFIX_LEN,
    SEMANTIC_BREAKPOINT_THRESHOLD,
    SEMANTIC_MAX_CHUNK_SIZE,
    SEMANTIC_MIN_CHUNK_SIZE,
)

from .base import BaseChunker, Chunk


class SemanticChunker(BaseChunker):
    """Group sentences into semantically coherent chunks using embedding similarity.

    For each pair of adjacent sentences, cosine similarity is computed between
    their embeddings.  When similarity drops below *breakpoint_threshold*, a new
    chunk is started.  Falls back to equal-size grouping when no ``embedding_fn``
    is supplied or the document has fewer than two sentences.
    """

    def __init__(
        self,
        embedding_fn: Any = None,
        breakpoint_threshold: float = SEMANTIC_BREAKPOINT_THRESHOLD,
        min_chunk_size: int = SEMANTIC_MIN_CHUNK_SIZE,
        max_chunk_size: int = SEMANTIC_MAX_CHUNK_SIZE,
    ) -> None:
        """Initialise the chunker.

        Args:
            embedding_fn: Callable ``(str) -> list[float]`` used to embed each sentence.
                When ``None``, falls back to equal-size grouping by *max_chunk_size*.
            breakpoint_threshold: Cosine similarity below which a boundary is inserted.
            min_chunk_size: Groups smaller than this are merged into the previous chunk.
            max_chunk_size: Groups that grow beyond this are flushed immediately.
        """
        self.embedding_fn = embedding_fn
        self.breakpoint_threshold = breakpoint_threshold
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size

    def chunk(self, text: str, metadata: dict[str, Any] | None = None) -> list[Chunk]:
        """Split *text* into semantically coherent chunks.

        Args:
            text: Source text to split.
            metadata: Passed through to every produced Chunk.

        Returns:
            List of Chunks grouped by semantic similarity.
        """
        metadata = metadata or {}
        doc_id = metadata.get("document_id", "doc")
        sentences = self._split_sentences(text)

        if not sentences:
            return []

        if self.embedding_fn is None or len(sentences) < 2:
            return self._fallback_chunk(sentences, text, doc_id, metadata)

        embeddings = self._embed_sentences(sentences)
        split_points = self._find_split_points(embeddings)
        groups = self._group_sentences(sentences, split_points)
        chunks: list[Chunk] = []
        for i, group in enumerate(groups):
            content = " ".join(group)
            start = text.find(content[:SEARCH_PREFIX_LEN])
            chunks.append(
                self._make_chunk(content, i, doc_id, start, start + len(content), metadata)
            )
        return chunks

    def _split_sentences(self, text: str) -> list[str]:
        """Tokenise *text* into individual sentences using punctuation markers.

        Args:
            text: Paragraph or document text.

        Returns:
            Non-empty sentence strings.
        """
        sentence_endings = re.compile(r"(?<=[.!?])\s+")
        raw = sentence_endings.split(text)
        return [s.strip() for s in raw if s.strip()]

    def _embed_sentences(self, sentences: list[str]) -> list[list[float]]:
        """Embed each sentence using the configured embedding function.

        Args:
            sentences: Tokenised sentences.

        Returns:
            Parallel list of embedding vectors.
        """
        return [self.embedding_fn(s) for s in sentences]

    def _cosine_similarity(self, a: list[float], b: list[float]) -> float:
        """Compute cosine similarity between two embedding vectors.

        Args:
            a: First embedding vector.
            b: Second embedding vector.

        Returns:
            Similarity in [0, 1]; 0.0 when either vector has zero norm.
        """
        va, vb = np.array(a), np.array(b)
        denom = np.linalg.norm(va) * np.linalg.norm(vb)
        return float(np.dot(va, vb) / denom) if denom > 0 else 0.0

    def _find_split_points(self, embeddings: list[list[float]]) -> list[int]:
        """Identify sentence indices where a semantic boundary should be placed.

        Args:
            embeddings: Embedding vector for each sentence.

        Returns:
            Sorted list of sentence indices where a new chunk should begin.
        """
        split_points: list[int] = []
        for i in range(len(embeddings) - 1):
            sim = self._cosine_similarity(embeddings[i], embeddings[i + 1])
            if sim < self.breakpoint_threshold:
                split_points.append(i + 1)
        return split_points

    def _group_sentences(
        self, sentences: list[str], split_points: list[int]
    ) -> list[list[str]]:
        """Collect sentences into groups separated at *split_points*.

        Groups smaller than *min_chunk_size* are merged into the previous group.
        Groups that exceed *max_chunk_size* are flushed immediately.

        Args:
            sentences: All sentences in the document.
            split_points: Sentence indices where a new group should begin.

        Returns:
            List of sentence groups, each group becoming one chunk.
        """
        groups: list[list[str]] = []
        current: list[str] = []
        for i, sentence in enumerate(sentences):
            if i in split_points and current:
                chunk_text = " ".join(current)
                if len(chunk_text) >= self.min_chunk_size:
                    groups.append(current)
                    current = []
                elif groups:
                    groups[-1].extend(current)
                    current = []
            current.append(sentence)
            if len(" ".join(current)) >= self.max_chunk_size:
                groups.append(current)
                current = []
        if current:
            groups.append(current)
        return groups

    def _fallback_chunk(
        self, sentences: list[str], text: str, doc_id: str, metadata: dict[str, Any]
    ) -> list[Chunk]:
        """Group sentences by *max_chunk_size* when no embedding function is available.

        Args:
            sentences: Tokenised sentences.
            text: Original source text (used for position lookup).
            doc_id: Parent document identifier.
            metadata: Metadata forwarded to each chunk.

        Returns:
            Chunks formed by accumulating sentences until *max_chunk_size* is reached.
        """
        chunks: list[Chunk] = []
        current: list[str] = []
        index = 0
        for sentence in sentences:
            current.append(sentence)
            if len(" ".join(current)) >= self.max_chunk_size:
                content = " ".join(current)
                start = text.find(content[:SEARCH_PREFIX_LEN])
                chunks.append(
                    self._make_chunk(content, index, doc_id, start, start + len(content), metadata)
                )
                index += 1
                current = []
        if current:
            content = " ".join(current)
            start = text.find(content[:SEARCH_PREFIX_LEN])
            chunks.append(
                self._make_chunk(content, index, doc_id, start, start + len(content), metadata)
            )
        return chunks
