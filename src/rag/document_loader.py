import logging
from dataclasses import dataclass
from typing import Any
import boto3
from botocore.exceptions import ClientError

logger = logging.getLogger(__name__)


@dataclass
class Document:
    content: str
    metadata: dict[str, Any]
    document_id: str


class S3DocumentLoader:
    """Load markdown documents from an S3 bucket."""

    def __init__(self, bucket_name: str, region: str = "us-east-1"):
        self.bucket_name = bucket_name
        self.region = region
        self._s3 = boto3.client("s3", region_name=region)

    def load_all(self, prefix: str = "", file_extensions: list[str] | None = None) -> list[Document]:
        extensions = file_extensions or [".md", ".txt"]
        keys = self._list_objects(prefix, extensions)
        documents = []
        for key in keys:
            try:
                doc = self.load_document(key)
                if doc:
                    documents.append(doc)
            except ClientError as e:
                logger.error("Failed to load s3://%s/%s: %s", self.bucket_name, key, e)
        logger.info("Loaded %d documents from s3://%s/%s", len(documents), self.bucket_name, prefix)
        return documents

    def load_document(self, key: str) -> Document | None:
        try:
            response = self._s3.get_object(Bucket=self.bucket_name, Key=key)
            content = response["Body"].read().decode("utf-8")
            metadata = {
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
        keys = []
        paginator = self._s3.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=self.bucket_name, Prefix=prefix):
            for obj in page.get("Contents", []):
                key = obj["Key"]
                if any(key.endswith(ext) for ext in extensions):
                    keys.append(key)
        return keys

    def upload_document(self, file_path: str, s3_key: str) -> bool:
        try:
            self._s3.upload_file(file_path, self.bucket_name, s3_key)
            logger.info("Uploaded %s to s3://%s/%s", file_path, self.bucket_name, s3_key)
            return True
        except ClientError as e:
            logger.error("Upload failed for %s: %s", file_path, e)
            return False
