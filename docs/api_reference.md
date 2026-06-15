# API Reference

## RAG System API

This document describes the API endpoints and interfaces for the RAG Document System.

## REST API Endpoints

### Query Endpoint

#### POST /query

Submit a question to the RAG system and receive a generated answer with source citations.

**Request Body:**
```json
{
  "question": "string",
  "chunking_strategy": "fixed_size | recursive | markdown | semantic | sentence | sliding_window | token",
  "top_k": 5,
  "filters": {
    "document_type": "string",
    "date_range": {
      "start": "ISO 8601 date",
      "end": "ISO 8601 date"
    }
  }
}
```

**Response:**
```json
{
  "answer": "string",
  "sources": [
    {
      "document_id": "string",
      "chunk_id": "string",
      "content": "string",
      "score": 0.95,
      "metadata": {
        "source": "s3://bucket/path/to/file.md",
        "section": "string"
      }
    }
  ],
  "chunking_strategy_used": "string",
  "processing_time_ms": 234
}
```

**Status Codes:**
- `200 OK`: Successful query
- `400 Bad Request`: Invalid request parameters
- `429 Too Many Requests`: Rate limit exceeded
- `500 Internal Server Error`: Server error

---

### Index Endpoint

#### POST /index

Trigger re-indexing of documents from S3.

**Request Body:**
```json
{
  "s3_prefix": "string",
  "chunking_strategy": "string",
  "force_reindex": false
}
```

**Response:**
```json
{
  "job_id": "string",
  "status": "started",
  "estimated_duration_seconds": 120
}
```

---

#### GET /index/{job_id}

Check the status of an indexing job.

**Response:**
```json
{
  "job_id": "string",
  "status": "running | completed | failed",
  "documents_processed": 45,
  "documents_total": 100,
  "chunks_created": 892,
  "errors": []
}
```

---

### Health Endpoint

#### GET /health

Check system health status.

**Response:**
```json
{
  "status": "healthy",
  "components": {
    "s3": "healthy",
    "vector_store": "healthy",
    "bedrock": "healthy",
    "embeddings": "healthy"
  },
  "version": "1.0.0"
}
```

---

## LangGraph Workflow API

### Available Chunking Strategies

| Strategy | ID | Description | Best Use Case |
|----------|----|----|---|
| Fixed Size | `fixed_size` | Split by character count | Uniform documents |
| Recursive | `recursive` | Multi-level text splitting | General purpose |
| Markdown | `markdown` | Split on MD headers | Technical docs |
| Semantic | `semantic` | Embedding-based grouping | Mixed content |
| Sentence | `sentence` | Sentence boundary splitting | Articles |
| Sliding Window | `sliding_window` | Overlapping chunks | Context continuity |
| Token | `token` | LLM token-based splitting | Precise context control |

### Workflow States

```
LOAD_DOCUMENTS → SELECT_CHUNKING → CHUNK_DOCUMENTS → EMBED_CHUNKS → 
STORE_VECTORS → RETRIEVE → GENERATE → END
```

## Bedrock Integration

### Supported Models

#### Text Generation
- `anthropic.claude-3-5-sonnet-20241022-v2:0` — Recommended
- `anthropic.claude-3-haiku-20240307-v1:0` — Fast, cost-effective
- `amazon.titan-text-express-v1` — AWS native

#### Embeddings
- `amazon.titan-embed-text-v2:0` — 1024 dimensions (recommended)
- `amazon.titan-embed-text-v1` — 1536 dimensions

### API Call Example

```python
import boto3
import json

bedrock = boto3.client('bedrock-runtime', region_name='us-east-1')

response = bedrock.invoke_model(
    modelId='anthropic.claude-3-5-sonnet-20241022-v2:0',
    body=json.dumps({
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 1024,
        "messages": [
            {
                "role": "user",
                "content": "Your question here"
            }
        ]
    })
)
```

## Environment Variables

| Variable | Description | Required | Default |
|----------|-------------|----------|---------|
| `AWS_REGION` | AWS region | Yes | `us-east-1` |
| `S3_BUCKET_NAME` | S3 bucket for documents | Yes | — |
| `OPENSEARCH_ENDPOINT` | OpenSearch endpoint URL | Yes | — |
| `BEDROCK_EMBED_MODEL` | Embedding model ID | No | `amazon.titan-embed-text-v2:0` |
| `BEDROCK_LLM_MODEL` | LLM model ID | No | `anthropic.claude-3-5-sonnet-20241022-v2:0` |
| `DEFAULT_CHUNK_SIZE` | Default chunk size in chars | No | `1000` |
| `DEFAULT_CHUNK_OVERLAP` | Default overlap in chars | No | `200` |
| `TOP_K_RESULTS` | Number of results to retrieve | No | `5` |
| `LOG_LEVEL` | Logging level | No | `INFO` |
