"""Context-passthrough generator for local development without Bedrock."""

from .generator import RAGResponse
from .vector_store import SearchResult


class LocalGenerator:
    """Return retrieved context directly as the answer — no LLM call.

    Drop-in replacement for RAGGenerator when running without AWS credentials.
    Same interface: generate(question, sources) -> RAGResponse.
    """

    def generate(self, question: str, sources: list[SearchResult]) -> RAGResponse:
        """Format retrieved chunks as the answer without calling an LLM.

        Args:
            question: The user's question (included in the answer header).
            sources: Retrieved chunks to surface as the answer body.

        Returns:
            RAGResponse whose answer is the formatted context text.
        """
        if not sources:
            answer = "No relevant documents found for your question."
        else:
            parts = [f"[LOCAL MODE — no LLM — showing retrieved context for: {question!r}]\n"]
            for i, result in enumerate(sources, 1):
                src = result.chunk.metadata.get("source", "unknown")
                parts.append(f"[{i}] {src} (score={result.score:.3f})\n{result.chunk.content}")
            answer = "\n\n---\n\n".join(parts)

        return RAGResponse(
            answer=answer,
            sources=sources,
            model_id="local",
            input_tokens=0,
            output_tokens=0,
        )
