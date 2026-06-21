#!/usr/bin/env python3
"""Prepare a Bedrock Agent and publish a versioned alias."""

import argparse
import logging
import sys
import time

import boto3
from botocore.exceptions import ClientError

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def prepare_agent(agent_id: str, region: str, max_wait: int = 120) -> bool:
    client = boto3.client("bedrock-agent", region_name=region)
    try:
        client.prepare_agent(agentId=agent_id)
        logger.info("Preparing agent %s…", agent_id)
    except ClientError as e:
        logger.error("Prepare failed: %s", e)
        return False

    waited = 0
    while waited < max_wait:
        resp = client.get_agent(agentId=agent_id)
        status = resp["agent"]["agentStatus"]
        if status == "PREPARED":
            logger.info("Agent prepared successfully")
            return True
        if "FAILED" in status:
            logger.error("Agent entered failed state: %s", status)
            return False
        time.sleep(5)
        waited += 5

    logger.error("Timed out waiting for agent to prepare")
    return False


def create_or_update_alias(agent_id: str, alias_name: str, region: str) -> str:
    client = boto3.client("bedrock-agent", region_name=region)

    # Check if alias exists
    paginator = client.get_paginator("list_agent_aliases")
    for page in paginator.paginate(agentId=agent_id):
        for alias in page.get("agentAliasSummaries", []):
            if alias["agentAliasName"] == alias_name:
                alias_id = alias["agentAliasId"]
                logger.info("Updating existing alias '%s' (%s)", alias_name, alias_id)
                client.update_agent_alias(
                    agentId=agent_id,
                    agentAliasId=alias_id,
                    agentAliasName=alias_name,
                    description="Updated by CI/CD pipeline",
                )
                return alias_id

    # Create new alias
    response = client.create_agent_alias(
        agentId=agent_id,
        agentAliasName=alias_name,
        description="Deployed via CI/CD pipeline",
    )
    alias_id = response["agentAlias"]["agentAliasId"]
    logger.info("Created alias '%s' (%s)", alias_name, alias_id)
    return alias_id


def main() -> int:
    parser = argparse.ArgumentParser(description="Deploy Bedrock Agent alias")
    parser.add_argument("--agent-id", required=True, help="Bedrock Agent ID")
    parser.add_argument("--knowledge-base-id", help="Associate KB if provided")
    parser.add_argument("--alias-name", default="prod", help="Alias name to publish")
    parser.add_argument("--region", default="us-east-1")
    args = parser.parse_args()

    if not prepare_agent(args.agent_id, args.region):
        return 1

    alias_id = create_or_update_alias(args.agent_id, args.alias_name, args.region)
    logger.info("Deployment complete. Agent: %s  Alias: %s", args.agent_id, alias_id)
    return 0


if __name__ == "__main__":
    sys.exit(main())
