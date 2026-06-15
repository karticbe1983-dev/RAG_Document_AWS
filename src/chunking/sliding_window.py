"""Sliding-window chunker with configurable window size, step, and boundary mode."""

from typing import Any

from config.settings import SLIDING_WINDOW_SIZE, SLIDING_STEP_SIZE, SEARCH_PREFIX_LEN
from .base import BaseChunker, Chunk


class SlidingWindowChunker(BaseChunker):
    """Fixed-size chunks with configurable stride to preserve context across boundaries.

    Two boundary modes are supported:

    - ``"char"`` — slide a character window of exactly *window_size* chars.
    - ``"word"`` — convert *window_size* and *step_size* to approximate word counts
      based on the document's average word length, then slice word groups.

    Use the word mode to avoid splitting mid-word.  Use the char mode for
    precise, deterministic chunk lengths.
    """

    def __init__(
        self,
        window_size: int = SLIDING_WINDOW_SIZE,
        step_size: int = SLIDING_STEP_SIZE,
        boundary: str = "word",
    ) -> None:
        """Initialise the chunker.

        Args:
            window_size: Characters (or approximate chars in word mode) per chunk.
            step_size: Characters to advance between consecutive chunk starts.
                Overlap = ``window_size - step_size``.
            boundary: ``"word"`` to respect word boundaries; ``"char"`` for exact slices.
        """
        self.window_size = window_size
        self.step_size = step_size
        self.boundary = boundary

    def chunk(self, text: str, metadata: dict[str, Any] | None = None) -> list[Chunk]:
        """Split *text* using a sliding window.

        Args:
            text: Source text to split.
            metadata: Passed through to every produced Chunk.

        Returns:
            Ordered list of overlapping Chunks.
        """
        metadata = metadata or {}
        doc_id = metadata.get("document_id", "doc")

        if self.boundary == "word":
            return self._word_boundary_chunk(text, doc_id, metadata)
        return self._char_chunk(text, doc_id, metadata)

    def _char_chunk(self, text: str, doc_id: str, metadata: dict[str, Any]) -> list[Chunk]:
        """Slide a character-level window across *text*.

        Args:
            text: Source text.
            doc_id: Parent document identifier.
            metadata: Metadata forwarded to each chunk.

        Returns:
            List of character-sliced Chunks.
        """
        chunks: list[Chunk] = []
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
        """Slide a word-group window across *text*, estimating words per window.

        The average word length of the document is used to convert *window_size*
        and *step_size* (in characters) into approximate word counts.

        Args:
            text: Source text.
            doc_id: Parent document identifier.
            metadata: Metadata forwarded to each chunk.

        Returns:
            List of word-group Chunks.
        """
        words = text.split()
        chunks: list[Chunk] = []
        index = 0
        avg_word_len = sum(len(w) for w in words) / max(len(words), 1) + 1
        words_per_window = max(1, int(self.window_size / avg_word_len))
        step_words = max(1, int(self.step_size / avg_word_len))

        i = 0
        while i < len(words):
            group = words[i : i + words_per_window]
            content = " ".join(group)
            start = text.find(content[:SEARCH_PREFIX_LEN]) if content else 0
            chunks.append(
                self._make_chunk(content, index, doc_id, start, start + len(content), metadata)
            )
            index += 1
            i += step_words

        return chunks
