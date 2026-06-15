import re
from typing import Any
from .base import BaseChunker, Chunk


class SentenceChunker(BaseChunker):
    """Group N sentences per chunk with optional overlap in sentences."""

    def __init__(
        self,
        sentences_per_chunk: int = 5,
        sentence_overlap: int = 1,
        min_sentence_length: int = 10,
    ):
        self.sentences_per_chunk = sentences_per_chunk
        self.sentence_overlap = sentence_overlap
        self.min_sentence_length = min_sentence_length

    def chunk(self, text: str, metadata: dict[str, Any] | None = None) -> list[Chunk]:
        metadata = metadata or {}
        doc_id = metadata.get("document_id", "doc")
        sentences = self._split_into_sentences(text)
        chunks = []
        index = 0
        step = max(1, self.sentences_per_chunk - self.sentence_overlap)

        i = 0
        while i < len(sentences):
            group = sentences[i : i + self.sentences_per_chunk]
            content = " ".join(group).strip()
            if content:
                start = text.find(content[:50]) if content else 0
                chunks.append(
                    self._make_chunk(content, index, doc_id, start, start + len(content), metadata)
                )
                index += 1
            i += step

        return chunks

    def _split_into_sentences(self, text: str) -> list[str]:
        # Split on sentence-ending punctuation followed by whitespace or end-of-string
        pattern = re.compile(r"(?<=[.!?])\s+|(?<=[.!?])$", re.MULTILINE)
        raw = pattern.split(text)
        return [
            s.strip()
            for s in raw
            if s.strip() and len(s.strip()) >= self.min_sentence_length
        ]
