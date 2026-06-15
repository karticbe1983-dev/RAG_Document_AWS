"""LangGraph node builder functions for every stage of the RAG pipeline.

Each ``build_*`` function is a factory that closes over its dependencies
(loader, embeddings, vector store, or generator) and returns a plain function
that accepts a ``RAGState`` dict and returns a partial-state update dict.
"""

import logging
from typing import Any

from config.settings import DEFAULT_TOP_K, CHUNK_PREVIEW_LEN, QUESTION_LOG_LEN
from .state import RAGState
from ..chunking.factory import ChunkingFactory, ChunkingStrategy
from ..rag.document_loader import S3DocumentLoader
from ..rag.embeddings import BedrockEmbeddings
from ..rag.retriever import RAGRetriever
from ..rag.generator import RAGGenerator
from ..rag.vector_store import OpenSearchVectorStore

logger = logging.getLogger(__name__)


def build_load_documents_node(
    loader: S3DocumentLoader,
) -> Any:
    """Return a node that loads documents from S3 into the pipeline state.

    Args:
        loader: Configured S3DocumentLoader instance.

    Returns:
        LangGraph-compatible node function.
    """

    def load_documents(state: RAGState) -> dict[str, Any]:
        """Fetch all documents from S3 and store them in state.

        Args:
            state: Current pipeline state.

        Returns:
            Partial state update with ``documents`` and ``processing_steps``.
        """
        logger.info("Loading documents from S3 (prefix=%s)", state.get("s3_prefix", ""))
        try:
            docs = loader.load_all(prefix=state.get("s3_prefix", ""))
            return {
                "documents": docs,
                "processing_steps": state.get("processing_steps", [])
                + [f"Loaded {len(docs)} documents from S3"],
            }
        except Exception as e:
            logger.error("Document loading failed: %s", e)
            return {"error": str(e), "documents": []}

    return load_documents


def build_chunk_documents_node(
    embeddings_fn: BedrockEmbeddings | None = None,
) -> Any:
    """Return a node that applies the state-selected chunking strategy.

    Args:
        embeddings_fn: Required only for the semantic strategy, which needs
            an embedding function at chunk time to compute sentence similarities.

    Returns:
        LangGraph-compatible node function.
    """

    def chunk_documents(state: RAGState) -> dict[str, Any]:
        """Split loaded documents into chunks using the strategy in state.

        Args:
            state: Current pipeline state; must contain ``documents`` and
                ``chunking_strategy``.

        Returns:
            Partial state update with ``chunks`` and ``processing_steps``.
        """
        strategy = state.get("chunking_strategy", ChunkingStrategy.RECURSIVE.value)
        documents = state.get("documents", [])
        logger.info("Chunking %d documents using strategy: %s", len(documents), strategy)

        chunker_kwargs: dict[str, Any] = {}
        if strategy == ChunkingStrategy.SEMANTIC.value and embeddings_fn:
            chunker_kwargs["embedding_fn"] = embeddings_fn

        chunker = ChunkingFactory.create(strategy, **chunker_kwargs)
        all_chunks = []
        for doc in documents:
            meta = {**doc.metadata, "document_id": doc.document_id}
            all_chunks.extend(chunker.chunk(doc.content, metadata=meta))

        return {
            "chunks": all_chunks,
            "processing_steps": state.get("processing_steps", [])
            + [f"Created {len(all_chunks)} chunks using '{strategy}'"],
        }

    return chunk_documents


def build_embed_chunks_node(embeddings: BedrockEmbeddings) -> Any:
    """Return a node that embeds all chunks in the pipeline state.

    Args:
        embeddings: Configured BedrockEmbeddings instance.

    Returns:
        LangGraph-compatible node function.
    """

    def embed_chunks(state: RAGState) -> dict[str, Any]:
        """Embed every chunk and store the vectors in state.

        Args:
            state: Current pipeline state; must contain ``chunks``.

        Returns:
            Partial state update with ``embeddings`` and ``processing_steps``.
        """
        chunks = state.get("chunks", [])
        logger.info("Embedding %d chunks", len(chunks))
        try:
            texts = [c.content for c in chunks]
            vecs = embeddings.embed_batch(texts)
            return {
                "embeddings": vecs,
                "processing_steps": state.get("processing_steps", [])
                + [f"Generated {len(vecs)} embeddings"],
            }
        except Exception as e:
            logger.error("Embedding failed: %s", e)
            return {"error": str(e), "embeddings": []}

    return embed_chunks


def build_store_vectors_node(vector_store: OpenSearchVectorStore) -> Any:
    """Return a node that writes chunks and their embeddings to OpenSearch.

    The node is a no-op when ``force_reindex`` is ``False`` in state.

    Args:
        vector_store: Configured OpenSearchVectorStore instance.

    Returns:
        LangGraph-compatible node function.
    """

    def store_vectors(state: RAGState) -> dict[str, Any]:
        """Index chunks into OpenSearch when force_reindex is True.

        Args:
            state: Current pipeline state; must contain ``chunks``, ``embeddings``,
                and ``force_reindex``.

        Returns:
            Partial state update with ``processing_steps``.
        """
        if not state.get("force_reindex", False):
            return {
                "processing_steps": state.get("processing_steps", [])
                + ["Skipped vector store update (force_reindex=False)"]
            }
        chunks = state.get("chunks", [])
        embeddings = state.get("embeddings", [])
        try:
            vector_store.create_index()
            added = vector_store.add_chunks(chunks, embeddings)
            return {
                "processing_steps": state.get("processing_steps", [])
                + [f"Stored {added} vectors in OpenSearch"]
            }
        except Exception as e:
            logger.error("Vector store error: %s", e)
            return {"error": str(e)}

    return store_vectors


def build_retrieve_node(retriever: RAGRetriever) -> Any:
    """Return a node that retrieves the top-k chunks for the user question.

    Args:
        retriever: Configured RAGRetriever instance.

    Returns:
        LangGraph-compatible node function.
    """

    def retrieve(state: RAGState) -> dict[str, Any]:
        """Embed the question and fetch the most relevant chunks from OpenSearch.

        Args:
            state: Current pipeline state; must contain ``question``, ``top_k``,
                and optionally ``filters``.

        Returns:
            Partial state update with ``search_results`` and ``processing_steps``.
        """
        question = state.get("question", "")
        top_k = state.get("top_k", DEFAULT_TOP_K)
        filters = state.get("filters") or None
        logger.info(
            "Retrieving top-%d chunks for question: %s",
            top_k,
            question[:QUESTION_LOG_LEN],
        )
        try:
            results = retriever.retrieve(question, top_k=top_k, filters=filters)
            return {
                "search_results": results,
                "processing_steps": state.get("processing_steps", [])
                + [f"Retrieved {len(results)} relevant chunks"],
            }
        except Exception as e:
            logger.error("Retrieval failed: %s", e)
            return {"error": str(e), "search_results": []}

    return retrieve


def build_generate_node(generator: RAGGenerator) -> Any:
    """Return a node that generates an answer from already-retrieved chunks.

    Args:
        generator: Configured RAGGenerator instance.

    Returns:
        LangGraph-compatible node function.
    """

    def generate(state: RAGState) -> dict[str, Any]:
        """Call the LLM with the question and the search_results already in state.

        Args:
            state: Current pipeline state; must contain ``question`` and
                ``search_results``.

        Returns:
            Partial state update with ``answer``, ``sources``, ``token_usage``,
            and ``processing_steps``.
        """
        question = state.get("question", "")
        results = state.get("search_results", [])
        logger.info("Generating answer from %d retrieved chunks", len(results))
        try:
            response = generator.generate(question, results)
            sources = [
                {
                    "content": r.chunk.content[:CHUNK_PREVIEW_LEN],
                    "score": r.score,
                    "source": r.chunk.metadata.get("source", ""),
                    "chunk_id": r.chunk.chunk_id,
                }
                for r in response.sources
            ]
            return {
                "answer": response.answer,
                "sources": sources,
                "token_usage": {
                    "input_tokens": response.input_tokens,
                    "output_tokens": response.output_tokens,
                },
                "processing_steps": state.get("processing_steps", [])
                + ["Generated answer with Bedrock Claude"],
            }
        except Exception as e:
            logger.error("Generation failed: %s", e)
            return {"error": str(e), "answer": ""}

    return generate


def check_for_errors(state: RAGState) -> str:
    """Conditional edge: route to the error handler when an error is recorded.

    Args:
        state: Current pipeline state.

    Returns:
        ``"error"`` if ``state["error"]`` is non-empty; ``"continue"`` otherwise.
    """
    return "error" if state.get("error") else "continue"
