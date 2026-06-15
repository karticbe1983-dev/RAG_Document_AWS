"""LLM generator: formats retrieved context and calls Bedrock Claude to produce an answer."""

import json
import logging
from dataclasses import dataclass

import boto3
from botocore.exceptions import ClientError

from config.settings import AWS_REGION, LLM_MODEL_ID, LLM_MAX_OUTPUT_TOKENS
from .vector_store import SearchResult

logger = logging.getLogger(__name__)

_RAG_PROMPT_TEMPLATE = """\
You are a knowledgeable assistant. Answer the user's question using ONLY the provided context.
If the context does not contain enough information to answer, say so clearly.

Context:
{context}

Question: {question}

Instructions:
- Answer based strictly on the provided context
- Cite specific sections when possible
- Be concise and accurate\
"""


@dataclass
class RAGResponse:
    """Result produced by RAGGenerator.generate()."""

    answer: str
    sources: list[SearchResult]
    model_id: str
    input_tokens: int
    output_tokens: int


class RAGGenerator:
    """Generate a grounded answer from retrieved chunks using a Bedrock Claude model.

    Responsibility: format context from SearchResults and call the LLM.
    Retrieval is handled separately by RAGRetriever.
    """

    def __init__(
        self,
        llm_model_id: str = LLM_MODEL_ID,
        region: str = AWS_REGION,
        max_output_tokens: int = LLM_MAX_OUTPUT_TOKENS,
    ) -> None:
        """Initialise the generator.

        Args:
            llm_model_id: Bedrock foundation model ID to use for generation.
            region: AWS region where the Bedrock endpoint is located.
            max_output_tokens: Maximum tokens the model may produce per call.
        """
        self.llm_model_id = llm_model_id
        self.max_output_tokens = max_output_tokens
        self._bedrock = boto3.client("bedrock-runtime", region_name=region)

    def generate(self, question: str, sources: list[SearchResult]) -> RAGResponse:
        """Format *sources* as context and ask the LLM to answer *question*.

        Args:
            question: The user's natural-language question.
            sources: Ranked list of retrieved chunks to use as grounding context.

        Returns:
            RAGResponse containing the answer text, source citations, and token counts.
        """
        context = self._format_context(sources)
        answer, input_tokens, output_tokens = self._call_llm(question, context)
        return RAGResponse(
            answer=answer,
            sources=sources,
            model_id=self.llm_model_id,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )

    def _format_context(self, results: list[SearchResult]) -> str:
        """Concatenate chunk content into a numbered, source-tagged context block.

        Args:
            results: Retrieved chunks with metadata.

        Returns:
            Multi-section string ready to be inserted into the prompt template.
        """
        parts = []
        for i, result in enumerate(results, 1):
            source = result.chunk.metadata.get("source", "unknown")
            parts.append(f"[{i}] Source: {source}\n{result.chunk.content}")
        return "\n\n---\n\n".join(parts)

    def _call_llm(self, question: str, context: str) -> tuple[str, int, int]:
        """Invoke the Bedrock model with the filled prompt template.

        Args:
            question: The user's question.
            context: Pre-formatted context string from _format_context.

        Returns:
            Tuple of (answer_text, input_token_count, output_token_count).

        Raises:
            ClientError: If the Bedrock API call fails.
        """
        prompt = _RAG_PROMPT_TEMPLATE.format(context=context, question=question)
        try:
            response = self._bedrock.invoke_model(
                modelId=self.llm_model_id,
                body=json.dumps({
                    "anthropic_version": "bedrock-2023-05-31",
                    "max_tokens": self.max_output_tokens,
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
