import logging
from typing import Any
from .state import RAGState
from ..chunking.factory import ChunkingFactory, ChunkingStrategy
from ..rag.document_loader import S3DocumentLoader
from ..rag.embeddings import BedrockEmbeddings
from ..rag.vector_store import OpenSearchVectorStore

logger = logging.getLogger(__name__)


def build_load_documents_node(loader: S3DocumentLoader):
    """Load markdown documents from S3."""

    def load_documents(state: RAGState) -> dict[str, Any]:
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


def build_chunk_documents_node(embeddings_fn: BedrockEmbeddings | None = None):
    """Apply the selected chunking strategy to loaded documents."""

    def chunk_documents(state: RAGState) -> dict[str, Any]:
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
            chunks = chunker.chunk(doc.content, metadata=meta)
            all_chunks.extend(chunks)

        return {
            "chunks": all_chunks,
            "processing_steps": state.get("processing_steps", [])
            + [f"Created {len(all_chunks)} chunks using '{strategy}'"],
        }

    return chunk_documents


def build_embed_chunks_node(embeddings: BedrockEmbeddings):
    """Embed all chunks using Bedrock Titan Embeddings."""

    def embed_chunks(state: RAGState) -> dict[str, Any]:
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


def build_store_vectors_node(vector_store: OpenSearchVectorStore):
    """Store chunk embeddings in OpenSearch (only when re-indexing)."""

    def store_vectors(state: RAGState) -> dict[str, Any]:
        chunks = state.get("chunks", [])
        embeddings = state.get("embeddings", [])
        if not state.get("force_reindex", False):
            return {
                "processing_steps": state.get("processing_steps", [])
                + ["Skipped vector store update (force_reindex=False)"]
            }
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


def build_retrieve_node(embeddings: BedrockEmbeddings, vector_store: OpenSearchVectorStore):
    """Retrieve relevant chunks for the user question."""

    def retrieve(state: RAGState) -> dict[str, Any]:
        question = state.get("question", "")
        top_k = state.get("top_k", 5)
        filters = state.get("filters") or None
        logger.info("Retrieving top-%d chunks for question: %s", top_k, question[:80])
        try:
            query_embedding = embeddings.embed(question)
            results = vector_store.similarity_search(query_embedding, top_k=top_k, filters=filters)
            return {
                "search_results": results,
                "processing_steps": state.get("processing_steps", [])
                + [f"Retrieved {len(results)} relevant chunks"],
            }
        except Exception as e:
            logger.error("Retrieval failed: %s", e)
            return {"error": str(e), "search_results": []}

    return retrieve


def build_generate_node(retriever):
    """Generate an answer using retrieved context and Bedrock Claude."""

    def generate(state: RAGState) -> dict[str, Any]:
        question = state.get("question", "")
        results = state.get("search_results", [])
        logger.info("Generating answer from %d retrieved chunks", len(results))
        try:
            response = retriever.query(question, top_k=len(results) or 5)
            sources = [
                {
                    "content": r.chunk.content[:200],
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


def route_by_chunking_strategy(state: RAGState) -> str:
    """Conditional edge: route to the appropriate chunking node label."""
    strategy = state.get("chunking_strategy", ChunkingStrategy.RECURSIVE.value)
    valid = ChunkingFactory.available_strategies()
    if strategy not in valid:
        logger.warning("Unknown strategy '%s', defaulting to recursive", strategy)
        return ChunkingStrategy.RECURSIVE.value
    return strategy


def check_for_errors(state: RAGState) -> str:
    """Conditional edge: stop pipeline if an error was recorded."""
    return "error" if state.get("error") else "continue"
