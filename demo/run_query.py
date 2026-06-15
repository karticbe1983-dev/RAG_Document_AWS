"""Demo: run a single RAG query with a chosen chunking strategy.

This file is a thin caller.  All logic lives in ``src/graph/workflow.py``.

Usage::

    python -m demo.run_query \\
        --bucket my-docs-bucket \\
        --opensearch-endpoint abc123.us-east-1.aoss.amazonaws.com \\
        --question "What is semantic chunking?" \\
        --strategy markdown
"""

import argparse
import os

from src.graph.workflow import RAGWorkflow, WorkflowConfig
from src.chunking.factory import ChunkingFactory


def main() -> None:
    """Parse CLI args and run one RAG query, printing the result."""
    parser = argparse.ArgumentParser(description="Run a single RAG query")
    parser.add_argument("--bucket", required=True)
    parser.add_argument("--opensearch-endpoint", required=True)
    parser.add_argument("--question", required=True)
    parser.add_argument(
        "--strategy",
        default="recursive",
        choices=ChunkingFactory.available_strategies(),
    )
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--s3-prefix", default="docs/")
    parser.add_argument("--force-reindex", action="store_true")
    args = parser.parse_args()

    config = WorkflowConfig(
        s3_bucket=args.bucket,
        opensearch_endpoint=args.opensearch_endpoint,
    )
    workflow = RAGWorkflow(config)

    result = workflow.run(
        question=args.question,
        chunking_strategy=args.strategy,
        top_k=args.top_k,
        s3_prefix=args.s3_prefix,
        force_reindex=args.force_reindex,
    )

    print(f"\nQuestion : {result['question']}")
    print(f"Strategy : {result['chunking_strategy']}")
    print(f"\nAnswer:\n{result['answer']}")
    print(f"\nSources ({len(result['sources'])}):")
    for i, src in enumerate(result["sources"], 1):
        print(f"  [{i}] {src['source']}  score={src['score']:.3f}")
    print(f"\nToken usage : {result['token_usage']}")
    print(f"Steps       : {' → '.join(result['processing_steps'])}")
    if result["error"]:
        print(f"\nERROR: {result['error']}")


if __name__ == "__main__":
    main()
