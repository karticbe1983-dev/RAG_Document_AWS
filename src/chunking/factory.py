"""Factory that instantiates the correct chunker for a given strategy name."""

from enum import StrEnum
from typing import Any

from config.settings import (
    FIXED_CHUNK_OVERLAP,
    FIXED_CHUNK_SIZE,
    MARKDOWN_CHUNK_OVERLAP,
    MARKDOWN_MAX_CHUNK_SIZE,
    RECURSIVE_CHUNK_OVERLAP,
    RECURSIVE_CHUNK_SIZE,
    SEMANTIC_BREAKPOINT_THRESHOLD,
    SEMANTIC_MAX_CHUNK_SIZE,
    SEMANTIC_MIN_CHUNK_SIZE,
    SENTENCE_OVERLAP,
    SENTENCE_PER_CHUNK,
    SLIDING_STEP_SIZE,
    SLIDING_WINDOW_SIZE,
    TOKEN_MAX_TOKENS,
    TOKEN_OVERLAP_TOKENS,
)

from .base import BaseChunker
from .fixed_size import FixedSizeChunker
from .markdown_aware import MarkdownChunker
from .recursive import RecursiveChunker
from .semantic import SemanticChunker
from .sentence import SentenceChunker
from .sliding_window import SlidingWindowChunker
from .token_based import TokenChunker


class ChunkingStrategy(StrEnum):
    """Canonical names for every supported chunking strategy."""

    FIXED_SIZE = "fixed_size"
    RECURSIVE = "recursive"
    MARKDOWN = "markdown"
    SEMANTIC = "semantic"
    SENTENCE = "sentence"
    SLIDING_WINDOW = "sliding_window"
    TOKEN = "token"


class ChunkingFactory:
    """Create chunkers by strategy name, applying settings-derived defaults.

    Callers can override any parameter via ``**kwargs``; unrecognised kwargs
    are silently forwarded to the constructor, so strategy-specific options
    (e.g. ``tokenizer`` for the token strategy) can be passed in a uniform way.
    """

    @staticmethod
    def create(strategy: "ChunkingStrategy | str", **kwargs: Any) -> BaseChunker:
        """Instantiate the chunker for *strategy*.

        Args:
            strategy: A ``ChunkingStrategy`` member or its string value.
            **kwargs: Optional overrides for chunker constructor parameters
                (e.g. ``chunk_size=500``, ``embedding_fn=my_embed``).

        Returns:
            A fully initialised ``BaseChunker`` subclass instance.

        Raises:
            ValueError: If *strategy* is not a recognised strategy name.
        """
        strategy = ChunkingStrategy(strategy)

        if strategy == ChunkingStrategy.FIXED_SIZE:
            return FixedSizeChunker(
                chunk_size=kwargs.get("chunk_size", FIXED_CHUNK_SIZE),
                chunk_overlap=kwargs.get("chunk_overlap", FIXED_CHUNK_OVERLAP),
            )
        if strategy == ChunkingStrategy.RECURSIVE:
            return RecursiveChunker(
                chunk_size=kwargs.get("chunk_size", RECURSIVE_CHUNK_SIZE),
                chunk_overlap=kwargs.get("chunk_overlap", RECURSIVE_CHUNK_OVERLAP),
                separators=kwargs.get("separators"),
            )
        if strategy == ChunkingStrategy.MARKDOWN:
            return MarkdownChunker(
                headers_to_split_on=kwargs.get("headers_to_split_on"),
                max_chunk_size=kwargs.get("max_chunk_size", MARKDOWN_MAX_CHUNK_SIZE),
                chunk_overlap=kwargs.get("chunk_overlap", MARKDOWN_CHUNK_OVERLAP),
            )
        if strategy == ChunkingStrategy.SEMANTIC:
            return SemanticChunker(
                embedding_fn=kwargs.get("embedding_fn"),
                breakpoint_threshold=kwargs.get("breakpoint_threshold", SEMANTIC_BREAKPOINT_THRESHOLD),
                min_chunk_size=kwargs.get("min_chunk_size", SEMANTIC_MIN_CHUNK_SIZE),
                max_chunk_size=kwargs.get("max_chunk_size", SEMANTIC_MAX_CHUNK_SIZE),
            )
        if strategy == ChunkingStrategy.SENTENCE:
            return SentenceChunker(
                sentences_per_chunk=kwargs.get("sentences_per_chunk", SENTENCE_PER_CHUNK),
                sentence_overlap=kwargs.get("sentence_overlap", SENTENCE_OVERLAP),
            )
        if strategy == ChunkingStrategy.SLIDING_WINDOW:
            return SlidingWindowChunker(
                window_size=kwargs.get("window_size", SLIDING_WINDOW_SIZE),
                step_size=kwargs.get("step_size", SLIDING_STEP_SIZE),
                boundary=kwargs.get("boundary", "word"),
            )
        if strategy == ChunkingStrategy.TOKEN:
            return TokenChunker(
                max_tokens=kwargs.get("max_tokens", TOKEN_MAX_TOKENS),
                overlap_tokens=kwargs.get("overlap_tokens", TOKEN_OVERLAP_TOKENS),
                tokenizer=kwargs.get("tokenizer"),
            )

        raise ValueError(f"Unknown chunking strategy: {strategy}")

    @staticmethod
    def available_strategies() -> list[str]:
        """Return the string values of every registered strategy.

        Returns:
            List of strategy name strings, e.g. ``["fixed_size", "recursive", ...]``.
        """
        return [s.value for s in ChunkingStrategy]
