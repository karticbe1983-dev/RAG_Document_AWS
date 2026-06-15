"""Hash-based deterministic embeddings for local development without Bedrock."""

import hashlib

import numpy as np

from config.settings import EMBED_DIMENSIONS


class LocalEmbeddings:
    """Produce deterministic pseudo-embeddings from text using word hashing.

    Each word is hashed (MD5) to a dimension index; the resulting sparse
    bag-of-words vector is L2-normalised.  Semantically similar texts share
    vocabulary and therefore produce similar vectors — good enough for local
    smoke-testing of the full pipeline.

    Drop-in replacement for BedrockEmbeddings: same embed/embed_batch/__call__ interface.
    """

    def __init__(self, dimensions: int = EMBED_DIMENSIONS) -> None:
        """Initialise the local embeddings.

        Args:
            dimensions: Output vector length; must match InMemoryVectorStore dimension.
        """
        self.dimensions = dimensions

    def embed(self, text: str) -> list[float]:
        """Hash every word in *text* into a normalised sparse vector.

        Args:
            text: Input text to embed.

        Returns:
            L2-normalised float list of length *dimensions*.
        """
        vec = np.zeros(self.dimensions, dtype=float)
        for word in text.lower().split():
            idx = int(hashlib.md5(word.encode()).hexdigest(), 16) % self.dimensions
            vec[idx] += 1.0
        norm = float(np.linalg.norm(vec))
        if norm > 0.0:
            vec /= norm
        return vec.tolist()

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of texts, one at a time.

        Args:
            texts: Input texts to embed.

        Returns:
            List of embedding vectors parallel to *texts*.
        """
        return [self.embed(t) for t in texts]

    def __call__(self, text: str) -> list[float]:
        """Allow this instance to be used as an embedding_fn callable.

        Args:
            text: Input text to embed.

        Returns:
            Embedding vector for *text*.
        """
        return self.embed(text)
