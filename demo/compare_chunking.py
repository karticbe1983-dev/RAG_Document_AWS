"""Demo: run the same question with every chunking strategy and compare chunk counts.

This file is a thin caller.  All logic lives in ``src/``.

Usage::

    python -m demo.compare_chunking \\
        --bucket my-docs-bucket \\
        --opensearch-endpoint abc123.us-east-1.aoss.amazonaws.com \\
        --question "Explain RAG and its benefits" \\
        --s3-prefix docs/
"""

import argparse

from src.chunking.factory import ChunkingFactory
from src.graph.workflow import RAGWorkflow, WorkflowConfig


def main() -> None:
    """Run every chunking strategy against the same question and print a comparison table."""
    parser = argparse.ArgumentParser(description="Compare all chunking strategies")
    parser.add_argument("--bucket", required=True)
    parser.add_argument("--opensearch-endpoint", required=True)
    parser.add_argument("--question", required=True)
    parser.add_argument("--s3-prefix", default="docs/")
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    config = WorkflowConfig(
        s3_bucket=args.bucket,
        opensearch_endpoint=args.opensearch_endpoint,
    )
    workflow = RAGWorkflow(config)

    strategies = ChunkingFactory.available_strategies()
    print(f"\nComparing {len(strategies)} strategies for question:\n  '{args.question}'\n")
    print(f"{'Strategy':<18} {'Steps':<50} {'Tokens In':>10} {'Tokens Out':>11}")
    print("-" * 93)

    for strategy in strategies:
        result = workflow.run(
            question=args.question,
            chunking_strategy=strategy,
            top_k=args.top_k,
            s3_prefix=args.s3_prefix,
            force_reindex=False,
        )
        usage = result["token_usage"]
        step_summary = " → ".join(result["processing_steps"])[:48]
        print(
            f"{strategy:<18} {step_summary:<50} "
            f"{usage.get('input_tokens', 0):>10} {usage.get('output_tokens', 0):>11}"
        )
        if result["error"]:
            print(f"  ERROR: {result['error']}")


if __name__ == "__main__":
    main()
