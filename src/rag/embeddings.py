"""Bedrock Titan Embeddings client: converts text into dense vector representations."""

import json
import logging
from typing import Any

import boto3
from botocore.exceptions import ClientError

from config.settings import AWS_REGION, EMBED_MODEL_ID, EMBED_DIMENSIONS, EMBED_BATCH_SIZE

logger = logging.getLogger(__name__)


class BedrockEmbeddings:
    """Generate text embeddings using Amazon Bedrock Titan Embeddings.

    Responsibility: call the Bedrock InvokeModel API and return embedding
    vectors.  No chunking, retrieval, or generation logic lives here.

    The ``__call__`` method makes instances usable as the ``embedding_fn``
    parameter accepted by ``SemanticChunker``.
    """

    def __init__(
        self,
        model_id: str = EMBED_MODEL_ID,
        region: str = AWS_REGION,
        dimensions: int = EMBED_DIMENSIONS,
    ) -> None:
        """Initialise the embeddings client.

        Args:
            model_id: Bedrock foundation model ID (must be a Titan Embeddings model).
            region: AWS region where the Bedrock endpoint is located.
            dimensions: Output vector dimension.  Only used by the v2 model.
        """
        self.model_id = model_id
        self.dimensions = dimensions
        self._client = boto3.client("bedrock-runtime", region_name=region)

    def embed(self, text: str) -> list[float]:
        """Embed a single text string and return its vector representation.

        Args:
            text: Input text to embed.

        Returns:
            List of floats representing the embedding vector.

        Raises:
            ClientError: If the Bedrock API call fails.
        """
        try:
            body = self._build_request_body(text)
            response = self._client.invoke_model(
                modelId=self.model_id,
                body=json.dumps(body),
                contentType="application/json",
                accept="application/json",
            )
            result = json.loads(response["body"].read())
            return result["embedding"]
        except ClientError as e:
            logger.error("Bedrock embedding error: %s", e)
            raise

    def embed_batch(
        self, texts: list[str], batch_size: int = EMBED_BATCH_SIZE
    ) -> list[list[float]]:
        """Embed multiple texts sequentially, respecting Bedrock rate limits.

        Args:
            texts: Input strings to embed.
            batch_size: Maximum texts to process per iteration (throttle guard).

        Returns:
            List of embedding vectors in the same order as *texts*.
        """
        embeddings: list[list[float]] = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            for text in batch:
                embeddings.append(self.embed(text))
        return embeddings

    def __call__(self, text: str) -> list[float]:
        """Alias for :meth:`embed` so instances can be passed as ``embedding_fn``.

        Args:
            text: Input text to embed.

        Returns:
            Embedding vector for *text*.
        """
        return self.embed(text)

    def _build_request_body(self, text: str) -> dict[str, Any]:
        """Construct the JSON body for the Bedrock InvokeModel call.

        Args:
            text: Text to embed.

        Returns:
            Dict matching the expected schema for the configured model.
        """
        if "titan-embed-text-v2" in self.model_id:
            return {"inputText": text, "dimensions": self.dimensions, "normalize": True}
        return {"inputText": text}
