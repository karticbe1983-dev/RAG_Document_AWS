"""Abstract base class and shared Chunk dataclass for all chunking strategies."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Chunk:
    """A single piece of text extracted from a document, with positional metadata."""

    content: str
    metadata: dict[str, Any] = field(default_factory=dict)
    chunk_id: str = ""
    document_id: str = ""
    start_index: int = 0
    end_index: int = 0


class BaseChunker(ABC):
    """Contract that every chunking strategy must satisfy."""

    @abstractmethod
    def chunk(self, text: str, metadata: dict[str, Any] | None = None) -> list[Chunk]:
        """Split *text* into a list of Chunk objects.

        Args:
            text: Raw document text to split.
            metadata: Key-value pairs attached to every produced chunk (e.g. source URL).

        Returns:
            Ordered list of Chunk objects covering *text*.
        """

    def _make_chunk(
        self,
        content: str,
        index: int,
        doc_id: str,
        start: int,
        end: int,
        metadata: dict[str, Any],
    ) -> Chunk:
        """Construct a Chunk with a deterministic ID derived from *doc_id* and *index*.

        Args:
            content: Text content of this chunk.
            index: Zero-based position of this chunk within the document.
            doc_id: Identifier of the parent document.
            start: Character offset where this chunk starts in the source text.
            end: Character offset where this chunk ends in the source text.
            metadata: Caller-supplied metadata; ``chunk_index`` is added automatically.

        Returns:
            A fully populated Chunk instance.
        """
        return Chunk(
            content=content,
            metadata={**metadata, "chunk_index": index},
            chunk_id=f"{doc_id}_chunk_{index}",
            document_id=doc_id,
            start_index=start,
            end_index=end,
        )
