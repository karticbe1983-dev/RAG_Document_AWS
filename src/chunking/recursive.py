"""Recursive character text splitter that honours natural text boundaries."""

from typing import Any

from config.settings import (
    RECURSIVE_CHUNK_SIZE,
    RECURSIVE_CHUNK_OVERLAP,
    RECURSIVE_SEPARATORS,
)
from .base import BaseChunker, Chunk


class RecursiveChunker(BaseChunker):
    """Recursively split text using a priority hierarchy of separators.

    Attempts separators in order (paragraphs → newlines → sentences → words →
    characters) and only falls back to a finer separator when the text still
    exceeds *chunk_size* after splitting on the current one.
    """

    def __init__(
        self,
        chunk_size: int = RECURSIVE_CHUNK_SIZE,
        chunk_overlap: int = RECURSIVE_CHUNK_OVERLAP,
        separators: list[str] | None = None,
    ) -> None:
        """Initialise the chunker.

        Args:
            chunk_size: Target maximum characters per chunk.
            chunk_overlap: Characters re-used at the start of each subsequent chunk.
            separators: Ordered list of separator strings to try; defaults to
                ``["\n\n", "\n", ". ", " ", ""]``.
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators if separators is not None else list(RECURSIVE_SEPARATORS)

    def chunk(self, text: str, metadata: dict[str, Any] | None = None) -> list[Chunk]:
        """Recursively split *text* into chunks that respect natural boundaries.

        Args:
            text: Source text to split.
            metadata: Passed through to every produced Chunk.

        Returns:
            List of Chunks; each is at most *chunk_size* characters when possible.
        """
        metadata = metadata or {}
        doc_id = metadata.get("document_id", "doc")
        raw_chunks = self._split(text, self.separators)
        chunks: list[Chunk] = []
        for i, content in enumerate(raw_chunks):
            start = text.find(content)
            chunks.append(
                self._make_chunk(content, i, doc_id, start, start + len(content), metadata)
            )
        return chunks

    def _split(self, text: str, separators: list[str]) -> list[str]:
        """Recursively apply separator hierarchy to produce sub-chunks.

        Args:
            text: Text remaining to be split.
            separators: Remaining separators to try, in priority order.

        Returns:
            List of text fragments each at most *chunk_size* chars.
        """
        if len(text) <= self.chunk_size:
            return [text]

        separator = separators[0] if separators else ""
        remaining_seps = separators[1:] if len(separators) > 1 else []

        if separator and separator in text:
            parts = text.split(separator)
        else:
            if remaining_seps:
                return self._split(text, remaining_seps)
            return self._force_split(text)

        chunks: list[str] = []
        current = ""

        for part in parts:
            candidate = current + (separator if current else "") + part
            if len(candidate) <= self.chunk_size:
                current = candidate
            else:
                if current:
                    chunks.append(current)
                if len(part) > self.chunk_size and remaining_seps:
                    chunks.extend(self._split(part, remaining_seps))
                    current = ""
                else:
                    current = part

        if current:
            chunks.append(current)

        return self._merge_with_overlap(chunks, separator)

    def _force_split(self, text: str) -> list[str]:
        """Slice *text* into fixed-size pieces when no separator works.

        Args:
            text: Text that cannot be split by any separator.

        Returns:
            List of raw character slices of length at most *chunk_size*.
        """
        return [
            text[i : i + self.chunk_size]
            for i in range(0, len(text), self.chunk_size - self.chunk_overlap)
        ]

    def _merge_with_overlap(self, chunks: list[str], separator: str) -> list[str]:
        """Re-join consecutive chunks so each one starts with the tail of the previous.

        Args:
            chunks: Freshly split fragments.
            separator: The separator string used to split them, re-inserted on merge.

        Returns:
            Fragments with *chunk_overlap* characters of context prepended to each.
        """
        if self.chunk_overlap == 0:
            return chunks
        merged: list[str] = []
        overlap_text = ""
        for chunk in chunks:
            combined = overlap_text + (separator if overlap_text else "") + chunk
            merged.append(combined)
            overlap_text = chunk[-self.chunk_overlap :] if len(chunk) > self.chunk_overlap else chunk
        return merged
