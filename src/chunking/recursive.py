from typing import Any
from .base import BaseChunker, Chunk


class RecursiveChunker(BaseChunker):
    """Recursively split text using a hierarchy of separators.

    Tries separators in order: paragraphs → sentences → words → characters.
    """

    DEFAULT_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        separators: list[str] | None = None,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or self.DEFAULT_SEPARATORS

    def chunk(self, text: str, metadata: dict[str, Any] | None = None) -> list[Chunk]:
        metadata = metadata or {}
        doc_id = metadata.get("document_id", "doc")
        raw_chunks = self._split(text, self.separators)
        chunks = []
        for i, content in enumerate(raw_chunks):
            start = text.find(content)
            chunks.append(
                self._make_chunk(content, i, doc_id, start, start + len(content), metadata)
            )
        return chunks

    def _split(self, text: str, separators: list[str]) -> list[str]:
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

        # Apply overlap by merging adjacent small chunks
        return self._merge_with_overlap(chunks, separator)

    def _force_split(self, text: str) -> list[str]:
        return [
            text[i : i + self.chunk_size]
            for i in range(0, len(text), self.chunk_size - self.chunk_overlap)
        ]

    def _merge_with_overlap(self, chunks: list[str], separator: str) -> list[str]:
        if self.chunk_overlap == 0:
            return chunks
        merged: list[str] = []
        overlap_text = ""
        for chunk in chunks:
            combined = overlap_text + (separator if overlap_text else "") + chunk
            merged.append(combined)
            # Keep last chunk_overlap chars as overlap for next chunk
            if len(chunk) > self.chunk_overlap:
                overlap_text = chunk[-self.chunk_overlap :]
            else:
                overlap_text = chunk
        return merged
