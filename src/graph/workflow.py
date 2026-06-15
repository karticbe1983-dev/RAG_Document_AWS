import logging
from dataclasses import dataclass
from typing import Any
from langgraph.graph import StateGraph, END
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

logger = logging.getLogger(__name__)

# Node names
NODE_LOAD = "load_documents"
NODE_ROUTE = "route_chunking"
NODE_EMBED = "embed_chunks"
NODE_STORE = "store_vectors"
NODE_RETRIEVE = "retrieve"
NODE_GENERATE = "generate"
NODE_ERROR = "handle_error"

# One dedicated node per chunking strategy
CHUNKING_NODES = {s.value: f"chunk_{s.value}" for s in ChunkingStrategy}


@dataclass
class WorkflowConfig:
    s3_bucket: str
    opensearch_endpoint: str
    opensearch_index: str = "rag-documents"
    aws_region: str = "us-east-1"
    embed_model_id: str = "amazon.titan-embed-text-v2:0"
    llm_model_id: str = "anthropic.claude-3-5-sonnet-20241022-v2:0"
    embed_dimensions: int = 1024
    default_top_k: int = 5


class RAGWorkflow:
    """LangGraph-powered RAG pipeline with a dedicated node per chunking strategy.

    Graph topology (simplified):
        load_documents
             ↓ (error check)
        route_chunking ──[conditional by strategy]──> chunk_<strategy>
                                                           ↓
                                                      embed_chunks
                                                           ↓ (error check)
                                                      store_vectors
                                                           ↓
                                                        retrieve
                                                           ↓ (error check)
                                                        generate
                                                           ↓
                                                          END
    Error paths at each stage route to handle_error → END.
    """

    def __init__(self, config: WorkflowConfig):
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
            llm_model_id=config.llm_model_id,
            region=config.aws_region,
            top_k=config.default_top_k,
        )
        self._graph = self._build_graph()

    # ──────────────────────────────────────────────────────────────────────
    # Graph construction
    # ──────────────────────────────────────────────────────────────────────

    def _build_graph(self) -> Any:
        graph = StateGraph(RAGState)

        # Fixed pipeline nodes
        graph.add_node(NODE_LOAD, build_load_documents_node(self._loader))
        graph.add_node(NODE_ROUTE, self._passthrough)
        graph.add_node(NODE_EMBED, build_embed_chunks_node(self._embeddings))
        graph.add_node(NODE_STORE, build_store_vectors_node(self._vector_store))
        graph.add_node(NODE_RETRIEVE, build_retrieve_node(self._embeddings, self._vector_store))
        graph.add_node(NODE_GENERATE, build_generate_node(self._retriever))
        graph.add_node(NODE_ERROR, self._handle_error)

        # One chunking node per strategy
        for strategy in ChunkingStrategy:
            node_name = CHUNKING_NODES[strategy.value]
            graph.add_node(node_name, self._make_strategy_node(strategy.value))
            graph.add_edge(node_name, NODE_EMBED)

        # Entry point
        graph.set_entry_point(NODE_LOAD)

        # load → (error check) → route_chunking
        graph.add_conditional_edges(
            NODE_LOAD,
            check_for_errors,
            {"error": NODE_ERROR, "continue": NODE_ROUTE},
        )

        # route_chunking → dispatch to strategy-specific chunking node
        graph.add_conditional_edges(
            NODE_ROUTE,
            self._select_chunking_node,
            {s.value: CHUNKING_NODES[s.value] for s in ChunkingStrategy},
        )

        # embed → (error check) → store_vectors
        graph.add_conditional_edges(
            NODE_EMBED,
            check_for_errors,
            {"error": NODE_ERROR, "continue": NODE_STORE},
        )

        # store_vectors → retrieve → (error check) → generate
        graph.add_edge(NODE_STORE, NODE_RETRIEVE)
        graph.add_conditional_edges(
            NODE_RETRIEVE,
            check_for_errors,
            {"error": NODE_ERROR, "continue": NODE_GENERATE},
        )

        graph.add_edge(NODE_GENERATE, END)
        graph.add_edge(NODE_ERROR, END)

        return graph.compile()

    # ──────────────────────────────────────────────────────────────────────
    # Node helpers
    # ──────────────────────────────────────────────────────────────────────

    def _make_strategy_node(self, strategy: str):
        """Wrap a chunk_documents call so it always uses the given strategy."""
        use_embeddings = (
            self._embeddings if strategy == ChunkingStrategy.SEMANTIC.value else None
        )
        chunk_node = build_chunk_documents_node(embeddings_fn=use_embeddings)

        def _node(state: RAGState) -> dict[str, Any]:
            patched = dict(state)
            patched["chunking_strategy"] = strategy
            return chunk_node(patched)  # type: ignore[arg-type]

        _node.__name__ = f"chunk_{strategy}"
        return _node

    @staticmethod
    def _passthrough(state: RAGState) -> dict[str, Any]:
        return {}

    @staticmethod
    def _handle_error(state: RAGState) -> dict[str, Any]:
        error = state.get("error", "Unknown error")
        logger.error("Workflow terminated with error: %s", error)
        return {
            "answer": f"Processing error: {error}",
            "sources": [],
        }

    @staticmethod
    def _select_chunking_node(state: RAGState) -> str:
        strategy = state.get("chunking_strategy", ChunkingStrategy.RECURSIVE.value)
        valid = {s.value for s in ChunkingStrategy}
        if strategy not in valid:
            logger.warning("Unknown strategy '%s', defaulting to recursive", strategy)
            return ChunkingStrategy.RECURSIVE.value
        return strategy

    # ──────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────

    def run(
        self,
        question: str,
        chunking_strategy: str = ChunkingStrategy.RECURSIVE.value,
        top_k: int = 5,
        s3_prefix: str = "",
        force_reindex: bool = False,
        filters: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Execute the full RAG pipeline and return the result."""
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

    def get_graph_diagram(self) -> str:
        """Return ASCII representation of the compiled graph."""
        try:
            return self._graph.get_graph().draw_ascii()
        except Exception:
            return "Graph diagram unavailable"
