# Retrieval Augmented Generation (RAG) Concepts

## What is RAG?

Retrieval Augmented Generation (RAG) is an AI framework that enhances large language model (LLM) outputs by incorporating information retrieved from external knowledge sources. Instead of relying solely on the model's training data, RAG systems dynamically fetch relevant information at query time.

## Why RAG?

### Problems RAG Solves
1. **Knowledge Cutoff**: LLMs have a training cutoff date and cannot access recent information
2. **Hallucination**: Without grounding, LLMs may generate plausible-sounding but incorrect information
3. **Domain Specificity**: General LLMs lack specialized knowledge about your organization or domain
4. **Context Length**: RAG selectively provides only the most relevant context, staying within token limits
5. **Cost**: Fine-tuning is expensive; RAG allows using the same model with different knowledge bases

## RAG Architecture Components

### 1. Document Ingestion Pipeline
The process of preparing documents for retrieval:
1. **Load**: Read documents from source (S3, databases, APIs)
2. **Parse**: Extract text from various formats (PDF, DOCX, HTML, Markdown)
3. **Chunk**: Split documents into manageable pieces
4. **Embed**: Convert text chunks to vector representations
5. **Store**: Index vectors in a vector database

### 2. Retrieval Pipeline
When a user query arrives:
1. **Embed Query**: Convert user question to vector
2. **Search**: Find similar vectors using k-NN search
3. **Rank**: Re-rank results by relevance (optional)
4. **Filter**: Apply metadata filters (optional)
5. **Return**: Provide top-k relevant chunks

### 3. Generation Pipeline
Combining retrieval with generation:
1. **Assemble Context**: Combine query + retrieved chunks
2. **Prompt Construction**: Format prompt with context and instructions
3. **LLM Generation**: Call the language model
4. **Post-processing**: Format and return response

## Chunking Strategies

Chunking is one of the most critical decisions in RAG system design.

### Fixed Size Chunking
Split text into chunks of fixed character or token count:
- **Pros**: Simple, predictable, fast
- **Cons**: May split mid-sentence or mid-concept
- **Best for**: Homogeneous documents, uniform content

### Recursive Character Text Splitting
Recursively splits using multiple separators (paragraphs → sentences → words):
- **Pros**: Respects natural text boundaries
- **Cons**: Chunk sizes can vary significantly
- **Best for**: General purpose text documents

### Markdown-Aware Chunking
Splits on markdown headers (##, ###):
- **Pros**: Maintains document structure and hierarchy
- **Cons**: Sections can be very long or very short
- **Best for**: Technical documentation, wikis

### Semantic Chunking
Groups sentences by semantic similarity using embeddings:
- **Pros**: Chunks are semantically coherent
- **Cons**: Computationally expensive, unpredictable chunk sizes
- **Best for**: When coherence is critical, mixed-topic documents

### Sentence-Based Chunking
Splits on sentence boundaries, groups N sentences:
- **Pros**: Maintains sentence integrity
- **Cons**: Less flexible chunk sizes
- **Best for**: News articles, academic papers

### Sliding Window Chunking
Fixed-size chunks with overlap between consecutive chunks:
- **Pros**: Ensures context continuity across chunk boundaries
- **Cons**: Redundant data increases storage
- **Best for**: Documents where context spans sections

### Token-Based Chunking
Splits based on LLM token count rather than characters:
- **Pros**: Precisely controls context window usage
- **Cons**: Requires tokenizer, slower
- **Best for**: When working close to context limits

## Embedding Models

### Dense Embeddings
- **Amazon Titan Embeddings v2**: 1024 dimensions, multilingual
- **Cohere Embed**: State-of-the-art retrieval performance
- **OpenAI text-embedding-ada-002**: Widely used baseline

### Sparse Embeddings (BM25)
Traditional TF-IDF based retrieval:
- Better for keyword matching
- No semantic understanding
- Fast and interpretable

### Hybrid Search
Combining dense and sparse retrieval:
- Reciprocal Rank Fusion (RRF) to merge results
- Better recall than either alone
- More complex to implement

## Advanced RAG Techniques

### Query Expansion
Generate multiple query variations to improve recall:
- HyDE (Hypothetical Document Embeddings)
- Multi-query retrieval
- Step-back prompting

### Re-ranking
Apply a second model to re-rank initial results:
- Cross-encoder models (more accurate than bi-encoders)
- Cohere Rerank
- ColBERT

### Context Compression
Reduce retrieved context to relevant portions:
- Extractive compression
- LLM-based compression
- Sentence scoring

### Self-RAG
Allow the LLM to decide when to retrieve:
- Generates retrieval tokens
- Evaluates relevance of retrieved documents
- Filters out irrelevant context

## Evaluation Metrics

### Retrieval Quality
- **Recall@k**: Fraction of relevant docs in top-k results
- **Precision@k**: Fraction of top-k results that are relevant
- **MRR**: Mean Reciprocal Rank
- **NDCG**: Normalized Discounted Cumulative Gain

### Generation Quality
- **Faithfulness**: Is the answer grounded in retrieved context?
- **Answer Relevance**: Does the answer address the question?
- **Context Relevance**: Is retrieved context relevant to the question?

### End-to-End Metrics
- **RAGAS**: RAG Assessment framework
- **TruLens**: Evaluation and tracking for LLM apps
- **Human evaluation**: Gold standard for quality assessment
