from typing import Any
from .base import BaseChunker, Chunk


class TokenChunker(BaseChunker):
    """Split text based on estimated token count.

    Uses a simple whitespace tokenizer by default. Pass a real tokenizer
    (e.g., tiktoken or a Hugging Face tokenizer) for precise splits.
    """

    def __init__(
        self,
        max_tokens: int = 512,
        overlap_tokens: int = 50,
        tokenizer: Any = None,
    ):
        self.max_tokens = max_tokens
        self.overlap_tokens = overlap_tokens
        self.tokenizer = tokenizer

    def chunk(self, text: str, metadata: dict[str, Any] | None = None) -> list[Chunk]:
        metadata = metadata or {}
        doc_id = metadata.get("document_id", "doc")

        tokens = self._tokenize(text)
        chunks = []
        index = 0
        step = max(1, self.max_tokens - self.overlap_tokens)

        i = 0
        while i < len(tokens):
            token_group = tokens[i : i + self.max_tokens]
            content = self._detokenize(token_group)
            start = text.find(content[:50]) if content else 0
            chunks.append(
                self._make_chunk(content, index, doc_id, start, start + len(content), metadata)
            )
            index += 1
            i += step

        return chunks

    def _tokenize(self, text: str) -> list[str]:
        if self.tokenizer:
            return self.tokenizer.encode(text)
        # Simple whitespace tokenization as fallback
        return text.split()

    def _detokenize(self, tokens: list[str]) -> str:
        if self.tokenizer and hasattr(self.tokenizer, "decode"):
            return self.tokenizer.decode(tokens)
        return " ".join(tokens)
