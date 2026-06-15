import json
import logging
from dataclasses import dataclass
from typing import Any
import boto3
from botocore.exceptions import ClientError
from .embeddings import BedrockEmbeddings
from .vector_store import OpenSearchVectorStore, SearchResult

logger = logging.getLogger(__name__)

CLAUDE_SONNET = "anthropic.claude-3-5-sonnet-20241022-v2:0"
CLAUDE_HAIKU = "anthropic.claude-3-haiku-20240307-v1:0"


@dataclass
class RAGResponse:
    answer: str
    sources: list[SearchResult]
    model_id: str
    input_tokens: int
    output_tokens: int


RAG_PROMPT_TEMPLATE = """You are a knowledgeable assistant. Answer the user's question using ONLY the provided context.
If the context does not contain enough information to answer, say so clearly.

Context:
{context}

Question: {question}

Instructions:
- Answer based strictly on the provided context
- Cite specific sections when possible
- Be concise and accurate"""


class RAGRetriever:
    """Orchestrate retrieval + generation for a single RAG query."""

    def __init__(
        self,
        embeddings: BedrockEmbeddings,
        vector_store: OpenSearchVectorStore,
        llm_model_id: str = CLAUDE_SONNET,
        region: str = "us-east-1",
        top_k: int = 5,
        max_tokens: int = 1024,
    ):
        self.embeddings = embeddings
        self.vector_store = vector_store
        self.llm_model_id = llm_model_id
        self.top_k = top_k
        self.max_tokens = max_tokens
        self._bedrock = boto3.client("bedrock-runtime", region_name=region)

    def query(
        self,
        question: str,
        filters: dict[str, Any] | None = None,
        top_k: int | None = None,
    ) -> RAGResponse:
        k = top_k or self.top_k
        query_embedding = self.embeddings.embed(question)
        sources = self.vector_store.similarity_search(query_embedding, top_k=k, filters=filters)
        context = self._format_context(sources)
        answer, input_tokens, output_tokens = self._generate(question, context)
        return RAGResponse(
            answer=answer,
            sources=sources,
            model_id=self.llm_model_id,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )

    def _format_context(self, results: list[SearchResult]) -> str:
        parts = []
        for i, result in enumerate(results, 1):
            source = result.chunk.metadata.get("source", "unknown")
            parts.append(f"[{i}] Source: {source}\n{result.chunk.content}")
        return "\n\n---\n\n".join(parts)

    def _generate(self, question: str, context: str) -> tuple[str, int, int]:
        prompt = RAG_PROMPT_TEMPLATE.format(context=context, question=question)
        try:
            response = self._bedrock.invoke_model(
                modelId=self.llm_model_id,
                body=json.dumps({
                    "anthropic_version": "bedrock-2023-05-31",
                    "max_tokens": self.max_tokens,
                    "messages": [{"role": "user", "content": prompt}],
                }),
                contentType="application/json",
                accept="application/json",
            )
            body = json.loads(response["body"].read())
            answer = body["content"][0]["text"]
            usage = body.get("usage", {})
            return answer, usage.get("input_tokens", 0), usage.get("output_tokens", 0)
        except ClientError as e:
            logger.error("Bedrock generation error: %s", e)
            raise
