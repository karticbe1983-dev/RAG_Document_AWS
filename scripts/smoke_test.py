#!/usr/bin/env python3
"""Smoke test: invoke the Bedrock Agent with a simple question and verify a response."""

import argparse
import logging
import sys
import uuid

import boto3
from botocore.exceptions import ClientError

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

SMOKE_QUESTION = "What is RAG and why is it useful?"
MIN_RESPONSE_LENGTH = 50


def get_prod_alias_id(agent_id: str, alias_name: str, region: str) -> str | None:
    client = boto3.client("bedrock-agent", region_name=region)
    paginator = client.get_paginator("list_agent_aliases")
    for page in paginator.paginate(agentId=agent_id):
        for alias in page.get("agentAliasSummaries", []):
            if alias["agentAliasName"] == alias_name:
                return alias["agentAliasId"]
    return None


def invoke_agent(agent_id: str, alias_id: str, question: str, region: str) -> str:
    runtime = boto3.client("bedrock-agent-runtime", region_name=region)
    session_id = str(uuid.uuid4())
    try:
        response = runtime.invoke_agent(
            agentId=agent_id,
            agentAliasId=alias_id,
            sessionId=session_id,
            inputText=question,
        )
        answer = ""
        for event in response.get("completion", []):
            if "chunk" in event:
                answer += event["chunk"]["bytes"].decode("utf-8")
        return answer.strip()
    except ClientError as e:
        logger.error("Invocation error: %s", e)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description="Smoke test the deployed Bedrock Agent")
    parser.add_argument("--agent-id", required=True)
    parser.add_argument("--alias-name", default="prod")
    parser.add_argument("--region", default="us-east-1")
    parser.add_argument("--question", default=SMOKE_QUESTION)
    args = parser.parse_args()

    alias_id = get_prod_alias_id(args.agent_id, args.alias_name, args.region)
    if not alias_id:
        logger.error("Alias '%s' not found for agent %s", args.alias_name, args.agent_id)
        return 1

    logger.info("Invoking agent %s (alias: %s)", args.agent_id, alias_id)
    logger.info("Question: %s", args.question)

    answer = invoke_agent(args.agent_id, alias_id, args.question, args.region)

    if len(answer) < MIN_RESPONSE_LENGTH:
        logger.error("Response too short (%d chars): %s", len(answer), answer)
        return 1

    logger.info("Response (%d chars):\n%s", len(answer), answer)
    logger.info("Smoke test PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
