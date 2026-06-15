import re
from typing import Any
import numpy as np
from .base import BaseChunker, Chunk


class SemanticChunker(BaseChunker):
    """Group sentences into semantically coherent chunks using embedding similarity.

    Computes cosine similarity between adjacent sentence embeddings and splits
    when similarity drops below a threshold.
    """

    def __init__(
        self,
        embedding_fn: Any = None,
        breakpoint_threshold: float = 0.75,
        min_chunk_size: int = 100,
        max_chunk_size: int = 2000,
    ):
        self.embedding_fn = embedding_fn
        self.breakpoint_threshold = breakpoint_threshold
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size

    def chunk(self, text: str, metadata: dict[str, Any] | None = None) -> list[Chunk]:
        metadata = metadata or {}
        doc_id = metadata.get("document_id", "doc")
        sentences = self._split_sentences(text)

        if not sentences:
            return []

        if self.embedding_fn is None or len(sentences) < 2:
            # Fallback: group into equal-size chunks
            return self._fallback_chunk(sentences, text, doc_id, metadata)

        embeddings = self._embed_sentences(sentences)
        split_points = self._find_split_points(embeddings)
        groups = self._group_sentences(sentences, split_points)
        chunks = []
        for i, group in enumerate(groups):
            content = " ".join(group)
            start = text.find(content[:50])
            chunks.append(
                self._make_chunk(content, i, doc_id, start, start + len(content), metadata)
            )
        return chunks

    def _split_sentences(self, text: str) -> list[str]:
        sentence_endings = re.compile(r"(?<=[.!?])\s+")
        raw = sentence_endings.split(text)
        return [s.strip() for s in raw if s.strip()]

    def _embed_sentences(self, sentences: list[str]) -> list[list[float]]:
        return [self.embedding_fn(s) for s in sentences]

    def _cosine_similarity(self, a: list[float], b: list[float]) -> float:
        va, vb = np.array(a), np.array(b)
        denom = np.linalg.norm(va) * np.linalg.norm(vb)
        return float(np.dot(va, vb) / denom) if denom > 0 else 0.0

    def _find_split_points(self, embeddings: list[list[float]]) -> list[int]:
        split_points = []
        for i in range(len(embeddings) - 1):
            sim = self._cosine_similarity(embeddings[i], embeddings[i + 1])
            if sim < self.breakpoint_threshold:
                split_points.append(i + 1)
        return split_points

    def _group_sentences(
        self, sentences: list[str], split_points: list[int]
    ) -> list[list[str]]:
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
        chunks = []
        current: list[str] = []
        index = 0
        for sentence in sentences:
            current.append(sentence)
            if len(" ".join(current)) >= self.max_chunk_size:
                content = " ".join(current)
                start = text.find(content[:50])
                chunks.append(
                    self._make_chunk(content, index, doc_id, start, start + len(content), metadata)
                )
                index += 1
                current = []
        if current:
            content = " ".join(current)
            start = text.find(content[:50])
            chunks.append(
                self._make_chunk(content, index, doc_id, start, start + len(content), metadata)
            )
        return chunks
