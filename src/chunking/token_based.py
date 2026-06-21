"""Token-count-based chunker with pluggable tokenizer support."""

from typing import Any

from config.settings import SEARCH_PREFIX_LEN, TOKEN_MAX_TOKENS, TOKEN_OVERLAP_TOKENS

from .base import BaseChunker, Chunk


class TokenChunker(BaseChunker):
    """Split text based on estimated token count with configurable overlap.

    By default, whitespace tokenization is used as a fast approximation.
    Pass any tokenizer that exposes ``encode(text) -> list`` and optionally
    ``decode(tokens) -> str`` (e.g. a ``tiktoken`` or Hugging Face tokenizer)
    for precise token-count splits aligned with the target LLM's vocabulary.
    """

    def __init__(
        self,
        max_tokens: int = TOKEN_MAX_TOKENS,
        overlap_tokens: int = TOKEN_OVERLAP_TOKENS,
        tokenizer: Any = None,
    ) -> None:
        """Initialise the chunker.

        Args:
            max_tokens: Maximum tokens (or whitespace words) per chunk.
            overlap_tokens: Tokens shared between consecutive chunks.
            tokenizer: Optional tokenizer with ``encode`` / ``decode`` methods.
                When ``None``, whitespace splitting is used.
        """
        self.max_tokens = max_tokens
        self.overlap_tokens = overlap_tokens
        self.tokenizer = tokenizer

    def chunk(self, text: str, metadata: dict[str, Any] | None = None) -> list[Chunk]:
        """Split *text* into chunks of at most *max_tokens* tokens.

        Args:
            text: Source text to split.
            metadata: Passed through to every produced Chunk.

        Returns:
            Ordered list of Chunks; consecutive chunks share *overlap_tokens* tokens.
        """
        metadata = metadata or {}
        doc_id = metadata.get("document_id", "doc")

        tokens = self._tokenize(text)
        chunks: list[Chunk] = []
        index = 0
        step = max(1, self.max_tokens - self.overlap_tokens)

        i = 0
        while i < len(tokens):
            token_group = tokens[i : i + self.max_tokens]
            content = self._detokenize(token_group)
            start = text.find(content[:SEARCH_PREFIX_LEN]) if content else 0
            chunks.append(
                self._make_chunk(content, index, doc_id, start, start + len(content), metadata)
            )
            index += 1
            i += step

        return chunks

    def _tokenize(self, text: str) -> list[str]:
        """Convert *text* to a token sequence using the configured tokenizer.

        Falls back to whitespace splitting when no tokenizer is set.

        Args:
            text: Input text.

        Returns:
            Sequence of token strings (or whitespace words).
        """
        if self.tokenizer:
            return self.tokenizer.encode(text)
        return text.split()

    def _detokenize(self, tokens: list[str]) -> str:
        """Reconstruct text from a token sequence.

        Uses the tokenizer's ``decode`` method when available, otherwise
        re-joins with spaces.

        Args:
            tokens: Token sequence to reconstruct.

        Returns:
            Reconstructed string.
        """
        if self.tokenizer and hasattr(self.tokenizer, "decode"):
            return self.tokenizer.decode(tokens)
        return " ".join(tokens)
