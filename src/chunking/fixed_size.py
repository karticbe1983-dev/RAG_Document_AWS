from typing import Any
from .base import BaseChunker, Chunk


class FixedSizeChunker(BaseChunker):
    """Split text into fixed-size character chunks with optional overlap."""

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk(self, text: str, metadata: dict[str, Any] | None = None) -> list[Chunk]:
        metadata = metadata or {}
        doc_id = metadata.get("document_id", "doc")
        chunks = []
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
