import json
import logging
import uuid
from dataclasses import dataclass
from typing import Any
import boto3
from opensearchpy import OpenSearch, RequestsHttpConnection, AWSV4SignerAuth
from botocore.auth import SigV4Auth
from botocore.awsrequest import AWSRequest
from ..chunking.base import Chunk

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    chunk: Chunk
    score: float


class OpenSearchVectorStore:
    """Store and retrieve chunk embeddings using Amazon OpenSearch Serverless."""

    INDEX_SETTINGS = {
        "settings": {"index": {"knn": True}},
        "mappings": {
            "properties": {
                "embedding": {
                    "type": "knn_vector",
                    "dimension": 1024,
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
        region: str = "us-east-1",
        dimension: int = 1024,
    ):
        self.endpoint = endpoint
        self.index_name = index_name
        self.region = region
        self.dimension = dimension
        self._client = self._build_client()

    def _build_client(self) -> OpenSearch:
        credentials = boto3.Session().get_credentials()
        auth = AWSV4SignerAuth(credentials, self.region, "aoss")
        return OpenSearch(
            hosts=[{"host": self.endpoint, "port": 443}],
            http_auth=auth,
            use_ssl=True,
            verify_certs=True,
            connection_class=RequestsHttpConnection,
        )

    def create_index(self, dimension: int | None = None) -> bool:
        settings = self.INDEX_SETTINGS.copy()
        if dimension:
            settings["mappings"]["properties"]["embedding"]["dimension"] = dimension
        if not self._client.indices.exists(self.index_name):
            self._client.indices.create(index=self.index_name, body=settings)
            logger.info("Created OpenSearch index: %s", self.index_name)
            return True
        return False

    def add_chunks(self, chunks: list[Chunk], embeddings: list[list[float]]) -> int:
        if len(chunks) != len(embeddings):
            raise ValueError("Chunks and embeddings must have the same length")

        added = 0
        for chunk, embedding in zip(chunks, embeddings):
            doc = {
                "embedding": embedding,
                "content": chunk.content,
                "document_id": chunk.document_id,
                "chunk_id": chunk.chunk_id or str(uuid.uuid4()),
                "metadata": chunk.metadata,
            }
            try:
                self._client.index(
                    index=self.index_name,
                    body=doc,
                    id=chunk.chunk_id or str(uuid.uuid4()),
                )
                added += 1
            except Exception as e:
                logger.error("Failed to index chunk %s: %s", chunk.chunk_id, e)

        logger.info("Indexed %d/%d chunks into %s", added, len(chunks), self.index_name)
        return added

    def similarity_search(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        filters: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        knn_query: dict[str, Any] = {
            "knn": {
                "embedding": {
                    "vector": query_embedding,
                    "k": top_k,
                }
            }
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

        results = []
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

    def delete_index(self) -> bool:
        if self._client.indices.exists(self.index_name):
            self._client.indices.delete(self.index_name)
            return True
        return False
