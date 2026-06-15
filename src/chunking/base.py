from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Chunk:
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)
    chunk_id: str = ""
    document_id: str = ""
    start_index: int = 0
    end_index: int = 0


class BaseChunker(ABC):
    @abstractmethod
    def chunk(self, text: str, metadata: dict[str, Any] | None = None) -> list[Chunk]:
        """Split text into chunks."""

    def _make_chunk(
        self,
        content: str,
        index: int,
        doc_id: str,
        start: int,
        end: int,
        metadata: dict[str, Any],
    ) -> Chunk:
        return Chunk(
            content=content,
            metadata={**metadata, "chunk_index": index},
            chunk_id=f"{doc_id}_chunk_{index}",
            document_id=doc_id,
            start_index=start,
            end_index=end,
        )
