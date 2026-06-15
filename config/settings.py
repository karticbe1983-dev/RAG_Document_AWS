"""Central configuration for every constant in the RAG Document system.

All magic numbers and magic strings are defined exactly once here and imported
everywhere they are used.  Change a value here to change it system-wide.
"""

# ── AWS ───────────────────────────────────────────────────────────────────────

AWS_REGION: str = "us-east-1"

# ── Model identifiers ─────────────────────────────────────────────────────────

EMBED_MODEL_ID: str = "amazon.titan-embed-text-v2:0"
LLM_MODEL_ID: str = "anthropic.claude-3-5-sonnet-20241022-v2:0"

# ── Embeddings ────────────────────────────────────────────────────────────────

EMBED_DIMENSIONS: int = 1024
"""Output vector size for amazon.titan-embed-text-v2:0."""

EMBED_BATCH_SIZE: int = 25
"""Maximum texts sent per Bedrock Embeddings call to avoid throttling."""

# ── Vector store (OpenSearch Serverless) ─────────────────────────────────────

OPENSEARCH_INDEX_NAME: str = "rag-index"
OPENSEARCH_PORT: int = 443

# ── Chunking — shared ─────────────────────────────────────────────────────────

SEARCH_PREFIX_LEN: int = 50
"""Chars taken from a chunk to locate it inside the source text with str.find()."""

# ── Chunking — FixedSize ─────────────────────────────────────────────────────

FIXED_CHUNK_SIZE: int = 1000
FIXED_CHUNK_OVERLAP: int = 200

# ── Chunking — Recursive ─────────────────────────────────────────────────────

RECURSIVE_CHUNK_SIZE: int = 1000
RECURSIVE_CHUNK_OVERLAP: int = 200
RECURSIVE_SEPARATORS: list[str] = ["\n\n", "\n", ". ", " ", ""]

# ── Chunking — Markdown ───────────────────────────────────────────────────────

MARKDOWN_MAX_CHUNK_SIZE: int = 2000
MARKDOWN_CHUNK_OVERLAP: int = 100
MARKDOWN_HEADERS: list[tuple[str, str]] = [
    ("#", "h1"),
    ("##", "h2"),
    ("###", "h3"),
    ("####", "h4"),
]

# ── Chunking — Semantic ───────────────────────────────────────────────────────

SEMANTIC_BREAKPOINT_THRESHOLD: float = 0.75
"""Cosine similarity below which two adjacent sentences are placed in separate chunks."""

SEMANTIC_MIN_CHUNK_SIZE: int = 100
SEMANTIC_MAX_CHUNK_SIZE: int = 2000

# ── Chunking — Sentence ───────────────────────────────────────────────────────

SENTENCE_PER_CHUNK: int = 5
SENTENCE_OVERLAP: int = 1
SENTENCE_MIN_LENGTH: int = 10
"""Minimum character length for a sentence to be included."""

# ── Chunking — SlidingWindow ─────────────────────────────────────────────────

SLIDING_WINDOW_SIZE: int = 1000
SLIDING_STEP_SIZE: int = 500

# ── Chunking — Token ─────────────────────────────────────────────────────────

TOKEN_MAX_TOKENS: int = 512
TOKEN_OVERLAP_TOKENS: int = 50

# ── Retrieval ─────────────────────────────────────────────────────────────────

DEFAULT_TOP_K: int = 5
"""Number of chunks returned by the vector store per query."""

HYBRID_SEARCH_ENABLED: bool = True
"""When True, retrieval blends knn vector search with BM25 keyword search."""

BM25_BOOST: float = 0.5
"""Relative weight of the BM25 (keyword) clause vs the knn (vector) clause.
Both clauses score on a 0–1 scale; this boost scales BM25 scores down so a
pure-keyword match doesn't dominate a strong semantic match.  Raise toward 1.0
to favour keyword precision; lower toward 0.1 to favour semantic recall."""

# ── Generation ────────────────────────────────────────────────────────────────

LLM_MAX_OUTPUT_TOKENS: int = 1024
"""Maximum tokens the LLM may generate in a single response."""

CHUNK_PREVIEW_LEN: int = 200
"""Characters of chunk content included in source citations."""

QUESTION_LOG_LEN: int = 80
"""Characters of the question written to the log line for brevity."""

# ── Bedrock Agent ─────────────────────────────────────────────────────────────

AGENT_IDLE_SESSION_TTL: int = 600
"""Seconds before an idle agent session expires."""

AGENT_PREPARE_TIMEOUT: int = 120
"""Seconds to wait for an agent to reach PREPARED status."""

AGENT_POLL_INTERVAL: int = 5
"""Seconds between status-poll calls when waiting for agent state transitions."""
