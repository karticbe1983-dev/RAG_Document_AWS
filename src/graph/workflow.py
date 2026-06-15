"""LangGraph RAG workflow: wires all pipeline nodes into a compiled state graph."""

import logging
from dataclasses import dataclass
from typing import Any

from langgraph.graph import StateGraph, END

from config.settings import (
    AWS_REGION,
    EMBED_MODEL_ID,
    LLM_MODEL_ID,
    EMBED_DIMENSIONS,
    DEFAULT_TOP_K,
    OPENSEARCH_INDEX_NAME,
)
from .state import RAGState
from .nodes import (
    build_load_documents_node,
    build_chunk_documents_node,
    build_embed_chunks_node,
    build_store_vectors_node,
    build_retrieve_node,
    build_generate_node,
    check_for_errors,
)
from ..chunking.factory import ChunkingStrategy
from ..rag.document_loader import S3DocumentLoader
from ..rag.embeddings import BedrockEmbeddings
from ..rag.vector_store import OpenSearchVectorStore
from ..rag.retriever import RAGRetriever
from ..rag.generator import RAGGenerator

logger = logging.getLogger(__name__)

# ── Node name constants ───────────────────────────────────────────────────────
_NODE_LOAD = "load_documents"
_NODE_ROUTE = "route_chunking"
_NODE_EMBED = "embed_chunks"
_NODE_STORE = "store_vectors"
_NODE_RETRIEVE = "retrieve"
_NODE_GENERATE = "generate"
_NODE_ERROR = "handle_error"

# One dedicated graph node per chunking strategy
_CHUNKING_NODES: dict[str, str] = {s.value: f"chunk_{s.value}" for s in ChunkingStrategy}


@dataclass
class WorkflowConfig:
    """All tuneable parameters for a RAGWorkflow instance."""

    s3_bucket: str
    """Name of the S3 bucket that holds the Markdown knowledge documents."""

    opensearch_endpoint: str
    """Hostname of the OpenSearch Serverless collection (no scheme or port)."""

    opensearch_index: str = OPENSEARCH_INDEX_NAME
    aws_region: str = AWS_REGION
    embed_model_id: str = EMBED_MODEL_ID
    llm_model_id: str = LLM_MODEL_ID
    embed_dimensions: int = EMBED_DIMENSIONS
    default_top_k: int = DEFAULT_TOP_K


class RAGWorkflow:
    """LangGraph-powered RAG pipeline with one dedicated node per chunking strategy.

    Graph topology::

        load_documents
             ↓ [check_for_errors]
        route_chunking ──[_select_chunking_node]──> chunk_<strategy>  (×7)
                                                          ↓
                                                    embed_chunks
                                                          ↓ [check_for_errors]
                                                    store_vectors
                                                          ↓
                                                       retrieve
                                                          ↓ [check_for_errors]
                                                       generate
                                                          ↓
                                                         END

    Error paths at every stage route to ``handle_error → END``.
    """

    def __init__(self, config: WorkflowConfig) -> None:
        """Initialise all service clients and compile the LangGraph state graph.

        Args:
            config: Workflow configuration including bucket name, OpenSearch endpoint,
                and model IDs.
        """
        self.config = config
        self._loader = S3DocumentLoader(config.s3_bucket, config.aws_region)
        self._embeddings = BedrockEmbeddings(
            model_id=config.embed_model_id,
            region=config.aws_region,
            dimensions=config.embed_dimensions,
        )
        self._vector_store = OpenSearchVectorStore(
            endpoint=config.opensearch_endpoint,
            index_name=config.opensearch_index,
            region=config.aws_region,
            dimension=config.embed_dimensions,
        )
        self._retriever = RAGRetriever(
            embeddings=self._embeddings,
            vector_store=self._vector_store,
        )
        self._generator = RAGGenerator(
            llm_model_id=config.llm_model_id,
            region=config.aws_region,
        )
        self._graph = self._build_graph()

    # ── Graph construction ────────────────────────────────────────────────────

    def _build_graph(self) -> Any:
        """Assemble and compile the LangGraph StateGraph.

        Returns:
            A compiled LangGraph graph ready to invoke.
        """
        graph = StateGraph(RAGState)

        graph.add_node(_NODE_LOAD, build_load_documents_node(self._loader))
        graph.add_node(_NODE_ROUTE, self._passthrough)
        graph.add_node(_NODE_EMBED, build_embed_chunks_node(self._embeddings))
        graph.add_node(_NODE_STORE, build_store_vectors_node(self._vector_store))
        graph.add_node(_NODE_RETRIEVE, build_retrieve_node(self._retriever))
        graph.add_node(_NODE_GENERATE, build_generate_node(self._generator))
        graph.add_node(_NODE_ERROR, self._handle_error)

        for strategy in ChunkingStrategy:
            node_name = _CHUNKING_NODES[strategy.value]
            graph.add_node(node_name, self._make_strategy_node(strategy.value))
            graph.add_edge(node_name, _NODE_EMBED)

        graph.set_entry_point(_NODE_LOAD)

        graph.add_conditional_edges(
            _NODE_LOAD,
            check_for_errors,
            {"error": _NODE_ERROR, "continue": _NODE_ROUTE},
        )
        graph.add_conditional_edges(
            _NODE_ROUTE,
            self._select_chunking_node,
            {s.value: _CHUNKING_NODES[s.value] for s in ChunkingStrategy},
        )
        graph.add_conditional_edges(
            _NODE_EMBED,
            check_for_errors,
            {"error": _NODE_ERROR, "continue": _NODE_STORE},
        )
        graph.add_edge(_NODE_STORE, _NODE_RETRIEVE)
        graph.add_conditional_edges(
            _NODE_RETRIEVE,
            check_for_errors,
            {"error": _NODE_ERROR, "continue": _NODE_GENERATE},
        )
        graph.add_edge(_NODE_GENERATE, END)
        graph.add_edge(_NODE_ERROR, END)

        return graph.compile()

    # ── Node helpers ──────────────────────────────────────────────────────────

    def _make_strategy_node(self, strategy: str) -> Any:
        """Create a chunking node that is hard-wired to *strategy*.

        The returned function patches ``chunking_strategy`` in state before
        delegating to ``build_chunk_documents_node`` so each strategy node
        is self-contained and the graph routing is fully declarative.

        Args:
            strategy: String value of the ``ChunkingStrategy`` to hard-wire.

        Returns:
            LangGraph-compatible node function.
        """
        use_embeddings = (
            self._embeddings if strategy == ChunkingStrategy.SEMANTIC.value else None
        )
        chunk_node = build_chunk_documents_node(embeddings_fn=use_embeddings)

        def _node(state: RAGState) -> dict[str, Any]:
            """Patch state with this node's strategy and delegate to chunk_node."""
            patched = dict(state)
            patched["chunking_strategy"] = strategy
            return chunk_node(patched)  # type: ignore[arg-type]

        _node.__name__ = f"chunk_{strategy}"
        return _node

    @staticmethod
    def _passthrough(state: RAGState) -> dict[str, Any]:
        """No-op routing node used as the dispatch point for chunking strategy selection.

        Args:
            state: Current pipeline state (unchanged).

        Returns:
            Empty dict (no state update).
        """
        return {}

    @staticmethod
    def _handle_error(state: RAGState) -> dict[str, Any]:
        """Terminal error node: log and surface the error as the answer.

        Args:
            state: Current pipeline state; ``error`` field holds the description.

        Returns:
            Partial state update with a human-readable ``answer`` and empty ``sources``.
        """
        error = state.get("error", "Unknown error")
        logger.error("Workflow terminated with error: %s", error)
        return {"answer": f"Processing error: {error}", "sources": []}

    @staticmethod
    def _select_chunking_node(state: RAGState) -> str:
        """Conditional edge function: map chunking_strategy in state to a node name.

        Falls back to ``recursive`` when the strategy value is unrecognised.

        Args:
            state: Current pipeline state.

        Returns:
            The string value of the target chunking strategy node.
        """
        strategy = state.get("chunking_strategy", ChunkingStrategy.RECURSIVE.value)
        valid = {s.value for s in ChunkingStrategy}
        if strategy not in valid:
            logger.warning("Unknown strategy '%s', defaulting to recursive", strategy)
            return ChunkingStrategy.RECURSIVE.value
        return strategy

    # ── Public API ────────────────────────────────────────────────────────────

    def run(
        self,
        question: str,
        chunking_strategy: str = ChunkingStrategy.RECURSIVE.value,
        top_k: int = DEFAULT_TOP_K,
        s3_prefix: str = "",
        force_reindex: bool = False,
        filters: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Execute the full RAG pipeline and return the structured result.

        Args:
            question: Natural-language question to answer.
            chunking_strategy: One of the ``ChunkingStrategy`` string values.
            top_k: Number of chunks to retrieve.
            s3_prefix: S3 key prefix to filter documents (e.g. ``"docs/"``).
            force_reindex: When ``True``, re-chunk, re-embed, and re-index all
                documents before retrieving.  Set ``False`` to use the existing index.
            filters: Optional OpenSearch term filters forwarded to the retriever.

        Returns:
            Dict with keys ``question``, ``answer``, ``sources``,
            ``chunking_strategy``, ``processing_steps``, ``token_usage``,
            and ``error``.
        """
        initial_state: RAGState = {
            "question": question,
            "chunking_strategy": chunking_strategy,
            "top_k": top_k,
            "filters": filters or {},
            "s3_prefix": s3_prefix,
            "force_reindex": force_reindex,
            "documents": [],
            "chunks": [],
            "embeddings": [],
            "search_results": [],
            "answer": "",
            "sources": [],
            "error": "",
            "processing_steps": [],
            "token_usage": {},
        }
        final_state = self._graph.invoke(initial_state)
        return {
            "question": question,
            "answer": final_state.get("answer", ""),
            "sources": final_state.get("sources", []),
            "chunking_strategy": chunking_strategy,
            "processing_steps": final_state.get("processing_steps", []),
            "token_usage": final_state.get("token_usage", {}),
            "error": final_state.get("error", ""),
        }
