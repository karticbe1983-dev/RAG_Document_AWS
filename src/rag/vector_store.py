"""OpenSearch Serverless vector store: index and search chunk embeddings."""

import logging
import uuid
from dataclasses import dataclass
from typing import Any, ClassVar

import boto3
from config.settings import AWS_REGION, BM25_BOOST, EMBED_DIMENSIONS, OPENSEARCH_PORT
from opensearchpy import AWSV4SignerAuth, OpenSearch, RequestsHttpConnection

from ..chunking.base import Chunk

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """A retrieved chunk paired with its similarity score."""

    chunk: Chunk
    score: float


class OpenSearchVectorStore:
    """Store chunk embeddings in Amazon OpenSearch Serverless and retrieve by k-NN.

    Responsibility: create the index, write chunk documents (with embeddings),
    and run k-NN similarity queries.  Embedding generation is handled by
    ``BedrockEmbeddings``; no embeddings are computed here.
    """

    _INDEX_SETTINGS: ClassVar[dict[str, Any]] = {
        "settings": {"index": {"knn": True}},
        "mappings": {
            "properties": {
                "embedding": {
                    "type": "knn_vector",
                    "dimension": EMBED_DIMENSIONS,
                    "method": {
                        "name": "hnsw",
                        "space_type": "cosinesimil",
                        "engine": "nmslib",
                    },
                },
                "content": {"type": "text"},
                "document_id": {"type": "keyword"},
                "chunk_id": {"type": "keyword"},
                "metadata": {"type": "object"},
            }
        },
    }

    def __init__(
        self,
        endpoint: str,
        index_name: str,
        region: str = AWS_REGION,
        dimension: int = EMBED_DIMENSIONS,
    ) -> None:
        """Initialise the vector store.

        Args:
            endpoint: OpenSearch Serverless collection endpoint hostname
                (without scheme or port).
            index_name: Name of the k-NN index to use.
            region: AWS region of the collection.
            dimension: Embedding vector dimension; must match the embedding model.
        """
        self.endpoint = endpoint
        self.index_name = index_name
        self.region = region
        self.dimension = dimension
        self._client = self._build_client()

    def create_index(self, dimension: int | None = None) -> bool:
        """Create the k-NN index if it does not already exist.

        Args:
            dimension: Override the embedding dimension stored in the class-level
                settings.  When ``None``, the constructor *dimension* is used.

        Returns:
            ``True`` if the index was created; ``False`` if it already existed.
        """
        settings = self._INDEX_SETTINGS.copy()
        actual_dim = dimension or self.dimension
        settings["mappings"]["properties"]["embedding"]["dimension"] = actual_dim
        if not self._client.indices.exists(self.index_name):
            self._client.indices.create(index=self.index_name, body=settings)
            logger.info("Created OpenSearch index: %s", self.index_name)
            return True
        return False

    def add_chunks(self, chunks: list[Chunk], embeddings: list[list[float]]) -> int:
        """Index a list of chunks together with their embedding vectors.

        Args:
            chunks: Chunk objects whose text content and metadata will be stored.
            embeddings: Parallel list of embedding vectors; must match *chunks* length.

        Returns:
            Number of chunks successfully indexed.

        Raises:
            ValueError: If *chunks* and *embeddings* have different lengths.
        """
        if len(chunks) != len(embeddings):
            raise ValueError("Chunks and embeddings must have the same length")

        added = 0
        for chunk, embedding in zip(chunks, embeddings, strict=False):
            chunk_id = chunk.chunk_id or str(uuid.uuid4())
            doc = {
                "embedding": embedding,
                "content": chunk.content,
                "document_id": chunk.document_id,
                "chunk_id": chunk_id,
                "metadata": chunk.metadata,
            }
            try:
                self._client.index(index=self.index_name, body=doc, id=chunk_id)
                added += 1
            except Exception as e:
                logger.error("Failed to index chunk %s: %s", chunk.chunk_id, e)

        logger.info("Indexed %d/%d chunks into %s", added, len(chunks), self.index_name)
        return added

    def similarity_search(
        self,
        query_embedding: list[float],
        top_k: int,
        filters: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        """Run a k-NN query and return the *top_k* most similar chunks.

        Args:
            query_embedding: Dense vector representing the query.
            top_k: Maximum number of results to return.
            filters: Optional dict of ``{field: value}`` term filters applied as
                a ``bool/filter`` wrapper around the k-NN query.

        Returns:
            List of SearchResult objects ordered by descending similarity score.
        """
        knn_query: dict[str, Any] = {
            "knn": {"embedding": {"vector": query_embedding, "k": top_k}}
        }
        if filters:
            knn_query = {
                "bool": {
                    "must": [knn_query],
                    "filter": [{"term": {k: v}} for k, v in filters.items()],
                }
            }

        response = self._client.search(
            index=self.index_name,
            body={"query": knn_query, "size": top_k},
        )

        results: list[SearchResult] = []
        for hit in response["hits"]["hits"]:
            source = hit["_source"]
            chunk = Chunk(
                content=source["content"],
                metadata=source.get("metadata", {}),
                chunk_id=source.get("chunk_id", ""),
                document_id=source.get("document_id", ""),
            )
            results.append(SearchResult(chunk=chunk, score=hit["_score"]))
        return results

    def hybrid_search(
        self,
        query_embedding: list[float],
        question: str,
        top_k: int,
        bm25_boost: float = BM25_BOOST,
        filters: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        """Run a hybrid knn + BM25 query and return the *top_k* most relevant chunks.

        Combines vector similarity (knn) with full-text keyword matching (BM25)
        in a single ``bool/should`` query.  OpenSearch sums the clause scores, so
        a chunk that ranks well on both axes scores higher than one that only ranks
        well on one.  *bm25_boost* controls the relative weight of keyword vs vector.

        Args:
            query_embedding: Dense vector for the query (same model as at index time).
            question: Raw question text used for the BM25 ``match`` clause.
            top_k: Number of results to return.
            bm25_boost: Boost multiplier applied to the BM25 clause score.
            filters: Optional ``{field: value}`` term filters applied as a
                ``bool/filter`` (zero-score, pre-filter semantics).

        Returns:
            List of SearchResult objects ordered by descending combined score.
        """
        query: dict[str, Any] = {
            "bool": {
                "should": [
                    {"knn": {"embedding": {"vector": query_embedding, "k": top_k}}},
                    {"match": {"content": {"query": question, "boost": bm25_boost}}},
                ],
                "minimum_should_match": 1,
            }
        }
        if filters:
            query["bool"]["filter"] = [
                {"term": {k: v}} for k, v in filters.items()
            ]

        response = self._client.search(
            index=self.index_name,
            body={"query": query, "size": top_k},
        )
        results: list[SearchResult] = []
        for hit in response["hits"]["hits"]:
            source = hit["_source"]
            chunk = Chunk(
                content=source["content"],
                metadata=source.get("metadata", {}),
                chunk_id=source.get("chunk_id", ""),
                document_id=source.get("document_id", ""),
            )
            results.append(SearchResult(chunk=chunk, score=hit["_score"]))
        return results

    def _build_client(self) -> OpenSearch:
        """Create and return an authenticated OpenSearch client for AOSS.

        Returns:
            Configured ``OpenSearch`` instance using AWS SigV4 signing.
        """
        credentials = boto3.Session().get_credentials()
        auth = AWSV4SignerAuth(credentials, self.region, "aoss")
        return OpenSearch(
            hosts=[{"host": self.endpoint, "port": OPENSEARCH_PORT}],
            http_auth=auth,
            use_ssl=True,
            verify_certs=True,
            connection_class=RequestsHttpConnection,
        )
