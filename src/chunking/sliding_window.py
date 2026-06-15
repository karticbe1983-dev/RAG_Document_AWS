from typing import Any
from .base import BaseChunker, Chunk


class SlidingWindowChunker(BaseChunker):
    """Fixed-size chunks with configurable overlap to preserve context across boundaries."""

    def __init__(
        self,
        window_size: int = 1000,
        step_size: int = 500,
        boundary: str = "word",
    ):
        self.window_size = window_size
        self.step_size = step_size
        self.boundary = boundary  # "word" | "char"

    def chunk(self, text: str, metadata: dict[str, Any] | None = None) -> list[Chunk]:
        metadata = metadata or {}
        doc_id = metadata.get("document_id", "doc")

        if self.boundary == "word":
            return self._word_boundary_chunk(text, doc_id, metadata)
        return self._char_chunk(text, doc_id, metadata)

    def _char_chunk(
        self, text: str, doc_id: str, metadata: dict[str, Any]
    ) -> list[Chunk]:
        chunks = []
        index = 0
        start = 0
        while start < len(text):
            end = min(start + self.window_size, len(text))
            content = text[start:end]
            chunks.append(self._make_chunk(content, index, doc_id, start, end, metadata))
            index += 1
            start += self.step_size
        return chunks

    def _word_boundary_chunk(
        self, text: str, doc_id: str, metadata: dict[str, Any]
    ) -> list[Chunk]:
        words = text.split()
        chunks = []
        index = 0
        # Estimate words per window based on average word length
        avg_word_len = sum(len(w) for w in words) / max(len(words), 1) + 1
        words_per_window = max(1, int(self.window_size / avg_word_len))
        step_words = max(1, int(self.step_size / avg_word_len))

        i = 0
        while i < len(words):
            group = words[i : i + words_per_window]
            content = " ".join(group)
            start = text.find(content[:50]) if content else 0
            chunks.append(
                self._make_chunk(content, index, doc_id, start, start + len(content), metadata)
            )
            index += 1
            i += step_words

        return chunks
