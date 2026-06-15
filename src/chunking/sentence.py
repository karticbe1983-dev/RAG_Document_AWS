"""Sentence-based chunker that groups N sentences per chunk with configurable overlap."""

import re
from typing import Any

from config.settings import (
    SENTENCE_PER_CHUNK,
    SENTENCE_OVERLAP,
    SENTENCE_MIN_LENGTH,
    SEARCH_PREFIX_LEN,
)
from .base import BaseChunker, Chunk


class SentenceChunker(BaseChunker):
    """Group N sentences per chunk with optional sentence-level overlap.

    Sentences are detected using punctuation-based regex.  Very short fragments
    (below *min_sentence_length* characters) are discarded before grouping.
    The stride between consecutive chunk start positions is
    ``sentences_per_chunk - sentence_overlap``, so each chunk shares
    *sentence_overlap* sentences with the next.
    """

    def __init__(
        self,
        sentences_per_chunk: int = SENTENCE_PER_CHUNK,
        sentence_overlap: int = SENTENCE_OVERLAP,
        min_sentence_length: int = SENTENCE_MIN_LENGTH,
    ) -> None:
        """Initialise the chunker.

        Args:
            sentences_per_chunk: How many sentences to include per chunk.
            sentence_overlap: Number of sentences shared between consecutive chunks.
            min_sentence_length: Fragments shorter than this are dropped.
        """
        self.sentences_per_chunk = sentences_per_chunk
        self.sentence_overlap = sentence_overlap
        self.min_sentence_length = min_sentence_length

    def chunk(self, text: str, metadata: dict[str, Any] | None = None) -> list[Chunk]:
        """Split *text* into chunks of *sentences_per_chunk* sentences each.

        Args:
            text: Source text to split.
            metadata: Passed through to every produced Chunk.

        Returns:
            Ordered list of Chunks.
        """
        metadata = metadata or {}
        doc_id = metadata.get("document_id", "doc")
        sentences = self._split_into_sentences(text)
        chunks: list[Chunk] = []
        index = 0
        step = max(1, self.sentences_per_chunk - self.sentence_overlap)

        i = 0
        while i < len(sentences):
            group = sentences[i : i + self.sentences_per_chunk]
            content = " ".join(group).strip()
            if content:
                start = text.find(content[:SEARCH_PREFIX_LEN])
                chunks.append(
                    self._make_chunk(content, index, doc_id, start, start + len(content), metadata)
                )
                index += 1
            i += step

        return chunks

    def _split_into_sentences(self, text: str) -> list[str]:
        """Tokenise *text* into sentences using punctuation-based regex.

        Sentences shorter than *min_sentence_length* characters are discarded.

        Args:
            text: Raw document or paragraph text.

        Returns:
            List of sentence strings, each at least *min_sentence_length* chars.
        """
        pattern = re.compile(r"(?<=[.!?])\s+|(?<=[.!?])$", re.MULTILINE)
        raw = pattern.split(text)
        return [
            s.strip()
            for s in raw
            if s.strip() and len(s.strip()) >= self.min_sentence_length
        ]
