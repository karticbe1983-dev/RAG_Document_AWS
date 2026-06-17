#!/usr/bin/env python3
"""Create the RAG vector index in OpenSearch Serverless.

Called by Terraform null_resource after the collection is created but before
the Bedrock Knowledge Base is provisioned (KB validates the index at creation).

Environment variables (injected by Terraform):
    OPENSEARCH_ENDPOINT  — collection endpoint, with or without https://
    AWS_REGION           — e.g. us-east-1
    INDEX_NAME           — defaults to rag-index
    EMBED_DIMENSIONS     — defaults to 1024
    COLLECTION_NAME      — collection name used to poll ACTIVE status
"""
import json
import os
import sys
import time

import boto3
from opensearchpy import OpenSearch, RequestsHttpConnection
from requests_aws4auth import AWS4Auth

REGION = os.environ.get("AWS_REGION", "us-east-1")
ENDPOINT = os.environ["OPENSEARCH_ENDPOINT"].removeprefix("https://")
INDEX_NAME = os.environ.get("INDEX_NAME", "rag-index")
DIMENSIONS = int(os.environ.get("EMBED_DIMENSIONS", "1024"))
COLLECTION_NAME = os.environ.get("COLLECTION_NAME", "rag-vectors")
MAX_WAIT = 300  # seconds


def wait_for_active() -> None:
    client = boto3.client("opensearchserverless", region_name=REGION)
    for elapsed in range(0, MAX_WAIT, 15):
        resp = client.batch_get_collection(names=[COLLECTION_NAME])
        details = resp.get("collectionDetails", [])
        if details and details[0].get("status") == "ACTIVE":
            print(f"Collection ACTIVE after ~{elapsed}s")
            return
        status = details[0].get("status", "NOT_FOUND") if details else "NOT_FOUND"
        print(f"  [{elapsed}s] status={status}, waiting 15s…")
        time.sleep(15)
    print("ERROR: timed out waiting for collection ACTIVE", file=sys.stderr)
    sys.exit(1)


def build_client() -> OpenSearch:
    session = boto3.Session()
    creds = session.get_credentials().get_frozen_credentials()
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
                        "parameters": {},
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
    wait_for_active()
    os_client = build_client()
    create_index(os_client)
