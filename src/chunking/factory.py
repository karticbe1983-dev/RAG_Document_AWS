from enum import Enum
from typing import Any
from .base import BaseChunker
from .fixed_size import FixedSizeChunker
from .recursive import RecursiveChunker
from .markdown_aware import MarkdownChunker
from .semantic import SemanticChunker
from .sentence import SentenceChunker
from .sliding_window import SlidingWindowChunker
from .token_based import TokenChunker


class ChunkingStrategy(str, Enum):
    FIXED_SIZE = "fixed_size"
    RECURSIVE = "recursive"
    MARKDOWN = "markdown"
    SEMANTIC = "semantic"
    SENTENCE = "sentence"
    SLIDING_WINDOW = "sliding_window"
    TOKEN = "token"


class ChunkingFactory:
    """Create chunkers by strategy name with consistent defaults."""

    @staticmethod
    def create(strategy: ChunkingStrategy | str, **kwargs: Any) -> BaseChunker:
        strategy = ChunkingStrategy(strategy)

        if strategy == ChunkingStrategy.FIXED_SIZE:
            return FixedSizeChunker(
                chunk_size=kwargs.get("chunk_size", 1000),
                chunk_overlap=kwargs.get("chunk_overlap", 200),
            )
        if strategy == ChunkingStrategy.RECURSIVE:
            return RecursiveChunker(
                chunk_size=kwargs.get("chunk_size", 1000),
                chunk_overlap=kwargs.get("chunk_overlap", 200),
                separators=kwargs.get("separators"),
            )
        if strategy == ChunkingStrategy.MARKDOWN:
            return MarkdownChunker(
                headers_to_split_on=kwargs.get("headers_to_split_on"),
                max_chunk_size=kwargs.get("max_chunk_size", 2000),
                chunk_overlap=kwargs.get("chunk_overlap", 100),
            )
        if strategy == ChunkingStrategy.SEMANTIC:
            return SemanticChunker(
                embedding_fn=kwargs.get("embedding_fn"),
                breakpoint_threshold=kwargs.get("breakpoint_threshold", 0.75),
                min_chunk_size=kwargs.get("min_chunk_size", 100),
                max_chunk_size=kwargs.get("max_chunk_size", 2000),
            )
        if strategy == ChunkingStrategy.SENTENCE:
            return SentenceChunker(
                sentences_per_chunk=kwargs.get("sentences_per_chunk", 5),
                sentence_overlap=kwargs.get("sentence_overlap", 1),
            )
        if strategy == ChunkingStrategy.SLIDING_WINDOW:
            return SlidingWindowChunker(
                window_size=kwargs.get("window_size", 1000),
                step_size=kwargs.get("step_size", 500),
                boundary=kwargs.get("boundary", "word"),
            )
        if strategy == ChunkingStrategy.TOKEN:
            return TokenChunker(
                max_tokens=kwargs.get("max_tokens", 512),
                overlap_tokens=kwargs.get("overlap_tokens", 50),
                tokenizer=kwargs.get("tokenizer"),
            )

        raise ValueError(f"Unknown chunking strategy: {strategy}")

    @staticmethod
    def available_strategies() -> list[str]:
        return [s.value for s in ChunkingStrategy]
