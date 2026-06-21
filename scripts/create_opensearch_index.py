#!/usr/bin/env python3
"""Create the RAG vector index in OpenSearch Serverless.

Called by Terraform null_resource after the collection is created but before
the Bedrock Knowledge Base is provisioned (KB validates the index at creation).

Environment variables (injected by Terraform):
    OPENSEARCH_ENDPOINT  — collection endpoint, with or without https://
    AWS_REGION           — e.g. us-east-1
    INDEX_NAME           — e.g. rag-index
    EMBED_DIMENSIONS     — defaults to 1024
"""
import json
import os
import sys

import boto3
from opensearchpy import OpenSearch, RequestsHttpConnection
from requests_aws4auth import AWS4Auth

REGION = os.environ.get("AWS_REGION", "us-east-1")
ENDPOINT = os.environ["OPENSEARCH_ENDPOINT"].removeprefix("https://")
INDEX_NAME = os.environ["INDEX_NAME"]
DIMENSIONS = int(os.environ.get("EMBED_DIMENSIONS", "1024"))


def build_client() -> OpenSearch:
    session = boto3.Session()
    raw_creds = session.get_credentials()
    if raw_creds is None:
        print("ERROR: no AWS credentials found", file=sys.stderr)
        sys.exit(1)
    creds = raw_creds.get_frozen_credentials()
    auth = AWS4Auth(
        creds.access_key,
        creds.secret_key,
        REGION,
        "aoss",
        session_token=creds.token,
    )
    return OpenSearch(
        hosts=[{"host": ENDPOINT, "port": 443}],
        http_auth=auth,
        use_ssl=True,
        verify_certs=True,
        connection_class=RequestsHttpConnection,
        timeout=30,
    )


def create_index(os_client: OpenSearch) -> None:
    if os_client.indices.exists(index=INDEX_NAME):
        print(f"Index '{INDEX_NAME}' already exists — skipping.")
        return

    body = {
        "settings": {"index.knn": True},
        "mappings": {
            "properties": {
                "embedding": {
                    "type": "knn_vector",
                    "dimension": DIMENSIONS,
                    "method": {
                        "engine": "faiss",
                        "space_type": "innerproduct",
                        "name": "hnsw",
                    },
                },
                "content": {"type": "text"},
                "metadata": {"type": "text"},
            }
        },
    }
    resp = os_client.indices.create(index=INDEX_NAME, body=body)
    print(f"Index '{INDEX_NAME}' created: {json.dumps(resp)}")


if __name__ == "__main__":
    os_client = build_client()
    create_index(os_client)
