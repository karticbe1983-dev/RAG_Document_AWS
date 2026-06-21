# RAG Document System — AWS Bedrock + LangGraph

A production-ready Retrieval-Augmented Generation (RAG) system built on AWS Bedrock, LangGraph, and OpenSearch Serverless.  
Documents live in S3 as Markdown files; a LangGraph state machine routes queries through **7 pluggable chunking strategies** before retrieving answers from Claude via Bedrock.  
A FastAPI layer (`src/api/`) exposes the full pipeline as an HTTP API and can run **entirely locally** without any AWS credentials via `LOCAL_MODE=true`.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         LangGraph Workflow                          │
│                                                                     │
│  S3 Docs ──► load_documents ──► route_chunking ──► chunk_<strategy>│
│                                        │                    │       │
│              ┌─────────────────────────┘                    ▼       │
│              │   fixed_size │ recursive │ markdown  │   embed_chunks│
│              │   semantic   │ sentence  │ sliding   │       │       │
│              │   token      └───────────────────────┘   store_vecs  │
│              │                                               │       │
│              └──────────────────────────────────────► retrieve      │
│                                                           │         │
│                                                       generate      │
│                                                     (Bedrock Claude)│
└─────────────────────────────────────────────────────────────────────┘
         │                                         │
    Amazon S3                            Amazon OpenSearch
   (Documents)                             Serverless
                                         (Vector Store)
```

### AWS Resources (managed by Terraform)

| Resource | Purpose |
|---|---|
| S3 Bucket (docs) | Source of truth for Markdown knowledge documents |
| S3 Bucket (artifacts) | Bedrock agent schemas and configs |
| OpenSearch Serverless | k-NN vector index for chunk embeddings |
| Bedrock Knowledge Base | Managed RAG pipeline with auto-ingestion |
| Bedrock Agent | Conversational agent backed by the Knowledge Base |
| IAM Roles | Bedrock, KB, and GitHub Actions OIDC roles |

---

## Chunking Strategies

| Strategy | Class | Best For |
|---|---|---|
| `fixed_size` | `FixedSizeChunker` | Uniform documents, simple baseline |
| `recursive` | `RecursiveChunker` | General purpose text (paragraphs → sentences → words) |
| `markdown_aware` | `MarkdownChunker` | Technical docs, wikis — splits on `#` headers |
| `semantic` | `SemanticChunker` | Mixed-topic docs — groups by embedding similarity |
| `sentence` | `SentenceChunker` | News articles, academic papers |
| `sliding_window` | `SlidingWindowChunker` | Context continuity across section boundaries |
| `token_based` | `TokenChunker` | Precise token-budget control for large contexts |

---

## Quick Start

### 1. Prerequisites

- Python 3.11+
- Terraform 1.6+
- AWS CLI configured with appropriate permissions
- AWS Bedrock access enabled for `anthropic.claude-3-5-sonnet-20241022-v2:0` and `amazon.titan-embed-text-v2:0`

### 2. Local development (no AWS required)

```bash
# Install dependencies
pip install -r requirements.txt -r requirements-dev.txt

# Run the API server in local mode (uses docs/ folder + in-memory vector store)
LOCAL_MODE=true uvicorn src.api.app:app --reload --port 8000

# Query it
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is RAG?", "chunking_strategy": "recursive"}'
```

### 3. Infrastructure (AWS)

```bash
# First time: create an S3 bucket for Terraform state
aws s3 mb s3://my-tfstate-bucket --region us-east-1

# Deploy all infrastructure
cd terraform
terraform init \
  -backend-config="bucket=my-tfstate-bucket" \
  -backend-config="key=rag-document-aws/dev/terraform.tfstate" \
  -backend-config="region=us-east-1"

terraform plan -var="environment=dev"
terraform apply -var="environment=dev"
```

> The `null_resource.create_vector_index` step runs automatically during `terraform apply` — it creates the knn_vector index in OpenSearch Serverless that Bedrock KB requires before it can be provisioned.

### 4. Upload documents

```bash
python scripts/upload_docs.py \
  --bucket $(terraform -chdir=terraform output -raw documents_bucket_name) \
  --local-dir docs/
```

### 5. Trigger Knowledge Base ingestion

```bash
python scripts/trigger_ingestion.py \
  --knowledge-base-id $(terraform -chdir=terraform output -raw knowledge_base_id) \
  --data-source-id $(terraform -chdir=terraform output -raw data_source_id)
```

### 6. Query the system programmatically

```python
from src.graph.workflow import RAGWorkflow, WorkflowConfig

config = WorkflowConfig(
    s3_bucket="your-docs-bucket",
    opensearch_endpoint="your-collection.us-east-1.aoss.amazonaws.com",
)
workflow = RAGWorkflow(config)

result = workflow.run(
    question="What chunking strategy is best for technical documentation?",
    chunking_strategy="markdown_aware",   # try any of the 7 strategies
    top_k=5,
    force_reindex=False,                  # set True to re-chunk and re-embed
)

print(result["answer"])
print("\nSources used:")
for s in result["sources"]:
    print(f"  - {s['source']} (score={s['score']:.3f})")
```

### 7. Run the API server against AWS

```bash
S3_BUCKET_NAME=rag-document-aws-docs-abfc23da \
OPENSEARCH_ENDPOINT=tkr2p52s19pgmwiv1lt1.us-east-1.aoss.amazonaws.com \
uvicorn src.api.app:app --reload --port 8000
```

---

## Project Layout

```
.
├── docs/                        # Markdown knowledge documents (synced to S3)
├── src/
│   ├── api/
│   │   └── app.py               # FastAPI server (LOCAL_MODE=true for no-AWS dev)
│   ├── chunking/                # 7 chunking strategies + factory
│   │   ├── base.py              # Chunk dataclass + BaseChunker ABC
│   │   ├── factory.py           # ChunkingFactory(strategy, **kwargs) → Chunker
│   │   ├── fixed_size.py
│   │   ├── recursive.py
│   │   ├── markdown_aware.py
│   │   ├── semantic.py
│   │   ├── sentence.py
│   │   ├── sliding_window.py
│   │   └── token_based.py
│   ├── rag/                     # AWS integrations + local dev equivalents
│   │   ├── document_loader.py       # S3DocumentLoader
│   │   ├── local_document_loader.py # Filesystem loader (LOCAL_MODE)
│   │   ├── embeddings.py            # BedrockEmbeddings (Titan v2)
│   │   ├── local_embeddings.py      # Hash-based deterministic embeddings (LOCAL_MODE)
│   │   ├── generator.py             # Bedrock Claude response generator
│   │   ├── local_generator.py       # Context-passthrough generator (LOCAL_MODE)
│   │   ├── vector_store.py          # OpenSearchVectorStore
│   │   ├── in_memory_vector_store.py# NumPy cosine-sim store (tests + LOCAL_MODE)
│   │   └── retriever.py             # RAGRetriever (embed query + search)
│   ├── graph/                   # LangGraph pipeline
│   │   ├── state.py                 # RAGState TypedDict
│   │   ├── nodes.py                 # Node builder functions
│   │   └── workflow.py              # RAGWorkflow (StateGraph)
│   └── agent/                   # Bedrock Agent management
│       └── bedrock_agent.py         # BedrockAgentDeployer
├── terraform/                   # All AWS infrastructure as code
│   ├── versions.tf              # Provider versions + S3 backend config
│   ├── variables.tf
│   ├── main.tf
│   ├── s3.tf
│   ├── opensearch.tf            # Collection + null_resource index bootstrap
│   ├── bedrock.tf               # Knowledge Base + Agent
│   ├── iam.tf
│   ├── outputs.tf
│   └── .terraform.lock.hcl     # Provider version lock (committed)
├── scripts/
│   ├── create_opensearch_index.py  # Bootstrap knn_vector index (called by Terraform)
│   ├── create_opensearch_index.sh  # Venv-aware wrapper for the above
│   ├── upload_docs.py              # Sync docs/ to S3
│   ├── trigger_ingestion.py        # Start KB ingestion job
│   ├── deploy_agent.py             # Prepare agent + publish alias
│   └── smoke_test.py               # Post-deploy health check
├── demo/
│   ├── compare_chunking.py      # Side-by-side strategy comparison
│   ├── invoke_agent.py          # Interactive agent session
│   └── run_query.py             # One-shot RAG query
├── tests/
│   ├── conftest.py              # Shared fixtures (mocked boto3, sample docs)
│   ├── test_chunking.py         # All 7 strategies + factory
│   ├── test_rag.py              # S3 loader, embeddings, retriever, generator
│   └── test_graph.py            # LangGraph nodes + routing
├── config/
│   └── settings.py              # Single source of truth for all constants
├── .github/workflows/
│   ├── deploy.yml               # Full CI/CD: test → plan → apply → sync → ingest → deploy
│   └── test.yml                 # PR gate: lint + type-check + unit tests
├── requirements.txt
├── requirements-dev.txt
├── pyproject.toml               # Ruff, mypy, pytest config
└── .env.example
```

---

## LOCAL_MODE

Setting `LOCAL_MODE=true` replaces all AWS dependencies with local equivalents:

| AWS component | Local replacement |
|---|---|
| S3 (documents) | `docs/` folder on disk |
| Bedrock Titan embeddings | Hash-based deterministic embeddings (`local_embeddings.py`) |
| Amazon OpenSearch Serverless | NumPy cosine-similarity store (`in_memory_vector_store.py`) |
| Bedrock Claude (generator) | Context passthrough — returns retrieved chunks as-is |

No credentials, no network, no cost. Useful for unit tests and rapid iteration on chunking logic.

---

## CI/CD Pipeline

```
Push to main
     │
     ├─► test          (pytest, ruff, mypy)
     ├─► terraform-plan
     ├─► terraform-apply  ──────────────────────────────────┐
     ├─► sync-documents   (aws s3 sync docs/ → S3)          │
     ├─► ingest-knowledge-base (Bedrock KB sync job)         │ Outputs: bucket,
     └─► deploy-agent   (prepare agent + alias) ◄───────────┘ KB ID, agent ID
              └─► smoke-test
```

Authentication uses GitHub Actions OIDC — no long-lived AWS credentials stored in GitHub Secrets.  
Only two secrets need to be configured in the repository:

| Secret | Value |
|---|---|
| `AWS_ROLE_ARN` | `arn:aws:iam::<account-id>:role/rag-document-aws-github-actions-role` |
| `TF_STATE_BUCKET` | Name of the S3 bucket holding Terraform state |

S3 document sync uses an explicit file-type allowlist (`*.md`, `*.pdf`, `*.txt`, `*.docx`, `*.csv`) — source code and secrets cannot reach the docs bucket even if accidentally present in `docs/`.

---

## Development

```bash
# Install dev dependencies
pip install -r requirements.txt -r requirements-dev.txt

# Run all unit tests (no AWS required)
pytest tests/ -m "not integration"

# Run a single test class
pytest tests/test_chunking.py::TestFixedSizeChunker -v

# Run integration tests (requires live AWS credentials + deployed infra)
pytest tests/ -m integration

# Lint + format
ruff check src/ tests/ config/ demo/ scripts/
ruff format src/ tests/ config/ demo/ scripts/

# Type check
mypy src/ config/

# Local API server
LOCAL_MODE=true uvicorn src.api.app:app --reload --port 8000

# Add a new chunking strategy:
# 1. Create src/chunking/my_strategy.py extending BaseChunker
# 2. Add MY_STRATEGY to ChunkingStrategy enum in factory.py
# 3. Add create() branch in ChunkingFactory
# 4. Add node in RAGWorkflow._build_graph()
# 5. Add tests in tests/test_chunking.py
```

---

## GitHub Repository Setup

```bash
# Create the remote repository (requires GitHub CLI)
gh repo create rag-document-aws --private --source=. --remote=origin

# Push
git push -u origin main

# Configure secrets
gh secret set AWS_ROLE_ARN --body "arn:aws:iam::<account-id>:role/rag-document-aws-github-actions-role"
gh secret set TF_STATE_BUCKET --body "my-tfstate-bucket"
```
