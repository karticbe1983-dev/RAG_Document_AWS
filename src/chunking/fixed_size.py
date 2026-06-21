"""Fixed-size character chunker with configurable overlap."""

from typing import Any

from config.settings import FIXED_CHUNK_OVERLAP, FIXED_CHUNK_SIZE

from .base import BaseChunker, Chunk


class FixedSizeChunker(BaseChunker):
    """Split text into fixed-size character chunks with optional overlap.

    The simplest chunking strategy: slide a window of *chunk_size* characters
    across the text, advancing by ``chunk_size - chunk_overlap`` each step.
    No attempt is made to align splits with word or sentence boundaries.
    """

    def __init__(
        self,
        chunk_size: int = FIXED_CHUNK_SIZE,
        chunk_overlap: int = FIXED_CHUNK_OVERLAP,
    ) -> None:
        """Initialise the chunker.

        Args:
            chunk_size: Maximum number of characters per chunk.
            chunk_overlap: Number of characters shared between consecutive chunks.
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk(self, text: str, metadata: dict[str, Any] | None = None) -> list[Chunk]:
        """Split *text* into fixed-size character chunks.

        Args:
            text: Source text to split.
            metadata: Passed through to every produced Chunk.

        Returns:
            List of Chunks whose content lengths are at most *chunk_size*.
        """
        metadata = metadata or {}
        doc_id = metadata.get("document_id", "doc")
        chunks: list[Chunk] = []
        start = 0
        index = 0

        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            content = text[start:end]
            chunks.append(self._make_chunk(content, index, doc_id, start, end, metadata))
            index += 1
            start += self.chunk_size - self.chunk_overlap
            if start >= len(text):
                break

        return chunks
