"""S3 document loader: reads Markdown and text files from an S3 bucket."""

import logging
from dataclasses import dataclass
from typing import Any

import boto3
from botocore.exceptions import ClientError
from config.settings import AWS_REGION

logger = logging.getLogger(__name__)

_DEFAULT_EXTENSIONS: list[str] = [".md", ".txt"]


@dataclass
class Document:
    """A single document loaded from S3, with its text content and provenance metadata."""

    content: str
    metadata: dict[str, Any]
    document_id: str


class S3DocumentLoader:
    """Load Markdown and text documents from a single S3 bucket.

    Responsibility: list S3 objects matching a prefix and extension filter,
    then fetch and decode each object into a Document.  No chunking or
    embedding is performed here.
    """

    def __init__(self, bucket_name: str, region: str = AWS_REGION) -> None:
        """Initialise the loader.

        Args:
            bucket_name: Name of the S3 bucket that holds the documents.
            region: AWS region where the bucket lives.
        """
        self.bucket_name = bucket_name
        self.region = region
        self._s3 = boto3.client("s3", region_name=region)

    def load_all(
        self,
        prefix: str = "",
        file_extensions: list[str] | None = None,
    ) -> list[Document]:
        """List every matching object under *prefix* and return them as Documents.

        Args:
            prefix: S3 key prefix to filter by (e.g. ``"docs/"``).
            file_extensions: Allowed file suffixes.  Defaults to ``[".md", ".txt"]``.

        Returns:
            All successfully loaded documents.  Objects that fail to load are
            logged and skipped.
        """
        extensions = file_extensions or _DEFAULT_EXTENSIONS
        keys = self._list_objects(prefix, extensions)
        documents: list[Document] = []
        for key in keys:
            try:
                doc = self.load_document(key)
                if doc:
                    documents.append(doc)
            except ClientError as e:
                logger.error("Failed to load s3://%s/%s: %s", self.bucket_name, key, e)
        logger.info(
            "Loaded %d documents from s3://%s/%s", len(documents), self.bucket_name, prefix
        )
        return documents

    def load_document(self, key: str) -> Document | None:
        """Fetch a single S3 object and return it as a Document.

        Args:
            key: S3 object key to fetch.

        Returns:
            Populated Document on success, or ``None`` if the object cannot be read.
        """
        try:
            response = self._s3.get_object(Bucket=self.bucket_name, Key=key)
            content = response["Body"].read().decode("utf-8")
            metadata: dict[str, Any] = {
                "source": f"s3://{self.bucket_name}/{key}",
                "key": key,
                "bucket": self.bucket_name,
                "content_type": response.get("ContentType", "text/markdown"),
                "last_modified": str(response.get("LastModified", "")),
                "size": response.get("ContentLength", 0),
                "document_id": key.replace("/", "_").replace(".", "_"),
            }
            return Document(
                content=content,
                metadata=metadata,
                document_id=metadata["document_id"],
            )
        except ClientError as e:
            logger.error("Error loading %s: %s", key, e)
            return None

    def _list_objects(self, prefix: str, extensions: list[str]) -> list[str]:
        """Page through S3 listing results and collect matching object keys.

        Args:
            prefix: S3 key prefix to filter by.
            extensions: File extensions to include (e.g. ``[".md"]``).

        Returns:
            Sorted list of S3 object keys that match the prefix and extension filter.
        """
        keys: list[str] = []
        paginator = self._s3.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=self.bucket_name, Prefix=prefix):
            for obj in page.get("Contents", []):
                key: str = obj["Key"]
                if any(key.endswith(ext) for ext in extensions):
                    keys.append(key)
        return keys
