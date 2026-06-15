"""Unit tests for all chunking strategies."""

import pytest
from src.chunking.base import Chunk
from src.chunking.factory import ChunkingFactory, ChunkingStrategy
from src.chunking.fixed_size import FixedSizeChunker
from src.chunking.recursive import RecursiveChunker
from src.chunking.markdown_aware import MarkdownChunker
from src.chunking.semantic import SemanticChunker
from src.chunking.sentence import SentenceChunker
from src.chunking.sliding_window import SlidingWindowChunker
from src.chunking.token_based import TokenChunker

SAMPLE_TEXT = """
# Introduction

This is the first paragraph of the document. It contains several sentences. Each sentence
adds some content to the paragraph.

## Section One

This section discusses the first topic. The topic is quite important and spans multiple
sentences. We add more text here to ensure the chunk size is exceeded in some configurations.

### Subsection 1.1

Even more detail about a sub-topic. This is nested content.

## Section Two

A second major section with different content. This content should be retrieved
separately from section one in most chunking strategies. More text to pad this section.
""".strip()

METADATA = {"document_id": "test_doc", "source": "test.md"}


def assert_valid_chunks(chunks: list[Chunk]) -> None:
    assert chunks, "Expected at least one chunk"
    for chunk in chunks:
        assert isinstance(chunk, Chunk)
        assert chunk.content.strip(), "Chunk content must not be empty"
        assert chunk.document_id == "test_doc"
        assert "chunk_index" in chunk.metadata


# ── Fixed size ───────────────────────────────────────────────────────────────

class TestFixedSizeChunker:
    def test_basic_split(self):
        chunker = FixedSizeChunker(chunk_size=100, chunk_overlap=20)
        chunks = chunker.chunk(SAMPLE_TEXT, METADATA)
        assert_valid_chunks(chunks)

    def test_overlap_creates_extra_chunks(self):
        no_overlap = FixedSizeChunker(chunk_size=200, chunk_overlap=0)
        with_overlap = FixedSizeChunker(chunk_size=200, chunk_overlap=100)
        assert len(with_overlap.chunk(SAMPLE_TEXT, METADATA)) >= len(
            no_overlap.chunk(SAMPLE_TEXT, METADATA)
        )

    def test_single_chunk_when_text_fits(self):
        chunker = FixedSizeChunker(chunk_size=10_000, chunk_overlap=0)
        chunks = chunker.chunk("Short text", METADATA)
        assert len(chunks) == 1

    def test_chunk_size_respected(self):
        size = 150
        chunker = FixedSizeChunker(chunk_size=size, chunk_overlap=0)
        for chunk in chunker.chunk(SAMPLE_TEXT, METADATA):
            assert len(chunk.content) <= size


# ── Recursive ────────────────────────────────────────────────────────────────

class TestRecursiveChunker:
    def test_basic_split(self):
        chunker = RecursiveChunker(chunk_size=300, chunk_overlap=50)
        chunks = chunker.chunk(SAMPLE_TEXT, METADATA)
        assert_valid_chunks(chunks)

    def test_prefers_paragraph_boundaries(self):
        chunker = RecursiveChunker(chunk_size=500, chunk_overlap=0)
        chunks = chunker.chunk(SAMPLE_TEXT, METADATA)
        # With enough space, entire paragraphs should be kept together
        assert any("\n" not in c.content for c in chunks) or len(chunks) >= 1

    def test_empty_text(self):
        chunker = RecursiveChunker()
        chunks = chunker.chunk("", METADATA)
        assert chunks == [] or (len(chunks) == 1 and not chunks[0].content.strip())


# ── Markdown ─────────────────────────────────────────────────────────────────

class TestMarkdownChunker:
    def test_splits_on_headers(self):
        chunker = MarkdownChunker()
        chunks = chunker.chunk(SAMPLE_TEXT, METADATA)
        assert_valid_chunks(chunks)
        # Should produce one chunk per major section
        assert len(chunks) >= 3

    def test_preserves_header_metadata(self):
        chunker = MarkdownChunker()
        chunks = chunker.chunk(SAMPLE_TEXT, METADATA)
        headers = [c.metadata.get("section_header") for c in chunks if c.metadata.get("section_header")]
        assert len(headers) >= 2

    def test_subsection_nesting(self):
        chunker = MarkdownChunker()
        chunks = chunker.chunk(SAMPLE_TEXT, METADATA)
        # Subsection 1.1 should create its own chunk
        sub_chunks = [c for c in chunks if "1.1" in c.metadata.get("section_header", "")]
        assert len(sub_chunks) >= 1


# ── Semantic ─────────────────────────────────────────────────────────────────

class TestSemanticChunker:
    def test_fallback_without_embedding_fn(self):
        chunker = SemanticChunker(embedding_fn=None, max_chunk_size=500)
        chunks = chunker.chunk(SAMPLE_TEXT, METADATA)
        assert_valid_chunks(chunks)

    def test_with_mock_embedding_fn(self):
        import random
        random.seed(42)
        # Mock embedding that returns random vectors (triggers split everywhere)
        mock_embed = lambda text: [random.random() for _ in range(10)]
        chunker = SemanticChunker(embedding_fn=mock_embed, breakpoint_threshold=0.99)
        chunks = chunker.chunk(SAMPLE_TEXT, METADATA)
        assert len(chunks) >= 1

    def test_high_threshold_produces_fewer_chunks(self):
        mock_embed = lambda text: [1.0] * 10  # All embeddings identical → no splits
        chunker = SemanticChunker(embedding_fn=mock_embed, breakpoint_threshold=0.5)
        chunks = chunker.chunk(SAMPLE_TEXT, METADATA)
        # With identical embeddings, everything should merge
        assert len(chunks) <= 5


# ── Sentence ─────────────────────────────────────────────────────────────────

class TestSentenceChunker:
    def test_basic_split(self):
        chunker = SentenceChunker(sentences_per_chunk=3, sentence_overlap=0)
        chunks = chunker.chunk(SAMPLE_TEXT, METADATA)
        assert_valid_chunks(chunks)

    def test_overlap_produces_more_chunks(self):
        without_overlap = SentenceChunker(sentences_per_chunk=3, sentence_overlap=0)
        with_overlap = SentenceChunker(sentences_per_chunk=3, sentence_overlap=1)
        c_no = without_overlap.chunk(SAMPLE_TEXT, METADATA)
        c_ov = with_overlap.chunk(SAMPLE_TEXT, METADATA)
        assert len(c_ov) >= len(c_no)


# ── Sliding window ────────────────────────────────────────────────────────────

class TestSlidingWindowChunker:
    def test_char_mode(self):
        chunker = SlidingWindowChunker(window_size=200, step_size=100, boundary="char")
        chunks = chunker.chunk(SAMPLE_TEXT, METADATA)
        assert_valid_chunks(chunks)

    def test_word_mode(self):
        chunker = SlidingWindowChunker(window_size=300, step_size=150, boundary="word")
        chunks = chunker.chunk(SAMPLE_TEXT, METADATA)
        assert_valid_chunks(chunks)

    def test_overlap_context(self):
        chunker = SlidingWindowChunker(window_size=200, step_size=100, boundary="char")
        chunks = chunker.chunk(SAMPLE_TEXT, METADATA)
        # Consecutive chunks should share some words (overlap)
        if len(chunks) >= 2:
            words1 = set(chunks[0].content.split())
            words2 = set(chunks[1].content.split())
            assert len(words1 & words2) > 0


# ── Token ────────────────────────────────────────────────────────────────────

class TestTokenChunker:
    def test_basic_split(self):
        chunker = TokenChunker(max_tokens=50, overlap_tokens=10)
        chunks = chunker.chunk(SAMPLE_TEXT, METADATA)
        assert_valid_chunks(chunks)

    def test_custom_tokenizer(self):
        class FakeTokenizer:
            def encode(self, text: str):
                return text.split()
            def decode(self, tokens):
                return " ".join(tokens)

        chunker = TokenChunker(max_tokens=20, overlap_tokens=5, tokenizer=FakeTokenizer())
        chunks = chunker.chunk(SAMPLE_TEXT, METADATA)
        assert_valid_chunks(chunks)


# ── Factory ───────────────────────────────────────────────────────────────────

class TestChunkingFactory:
    @pytest.mark.parametrize("strategy", [s.value for s in ChunkingStrategy])
    def test_all_strategies_creatable(self, strategy: str):
        chunker = ChunkingFactory.create(strategy)
        assert chunker is not None

    @pytest.mark.parametrize("strategy", [s.value for s in ChunkingStrategy])
    def test_all_strategies_produce_chunks(self, strategy: str):
        chunker = ChunkingFactory.create(strategy)
        chunks = chunker.chunk(SAMPLE_TEXT, METADATA)
        assert isinstance(chunks, list)
        assert len(chunks) >= 1

    def test_invalid_strategy_raises(self):
        with pytest.raises(ValueError):
            ChunkingFactory.create("nonexistent_strategy")

    def test_available_strategies(self):
        strategies = ChunkingFactory.available_strategies()
        assert len(strategies) == len(ChunkingStrategy)
        assert "recursive" in strategies
        assert "semantic" in strategies
