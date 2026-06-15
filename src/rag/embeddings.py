import json
import logging
from typing import Any
import boto3
from botocore.exceptions import ClientError

logger = logging.getLogger(__name__)

TITAN_EMBED_V2 = "amazon.titan-embed-text-v2:0"
TITAN_EMBED_V1 = "amazon.titan-embed-text-v1"


class BedrockEmbeddings:
    """Generate text embeddings using Amazon Bedrock Titan Embeddings."""

    def __init__(
        self,
        model_id: str = TITAN_EMBED_V2,
        region: str = "us-east-1",
        dimensions: int = 1024,
    ):
        self.model_id = model_id
        self.dimensions = dimensions
        self._client = boto3.client("bedrock-runtime", region_name=region)

    def embed(self, text: str) -> list[float]:
        """Embed a single text string."""
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

    def embed_batch(self, texts: list[str], batch_size: int = 25) -> list[list[float]]:
        """Embed multiple texts, respecting Bedrock rate limits."""
        embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            for text in batch:
                embeddings.append(self.embed(text))
        return embeddings

    def __call__(self, text: str) -> list[float]:
        return self.embed(text)

    def _build_request_body(self, text: str) -> dict[str, Any]:
        if "titan-embed-text-v2" in self.model_id:
            return {
                "inputText": text,
                "dimensions": self.dimensions,
                "normalize": True,
            }
        # Titan Embeddings v1 format
        return {"inputText": text}
