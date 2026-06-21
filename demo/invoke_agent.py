"""Demo: deploy a Bedrock Agent and send it a test question.

This file is a thin caller.  All agent management logic lives in
``src/agent/bedrock_agent.py``.

Usage::

    python -m demo.invoke_agent \\
        --agent-name rag-document-agent \\
        --role-arn arn:aws:iam::123456789:role/bedrock-agent-role \\
        --knowledge-base-id ABCDEFGH12 \\
        --alias-name prod \\
        --question "What chunking strategy works best for technical docs?"
"""

import argparse
import uuid

import boto3
from config.settings import AWS_REGION
from src.agent.bedrock_agent import AgentConfig, BedrockAgentDeployer


def main() -> None:
    """Create-or-update the agent, associate a KB, prepare it, publish an alias, then invoke it."""
    parser = argparse.ArgumentParser(description="Deploy and invoke a Bedrock Agent")
    parser.add_argument("--agent-name", default="rag-document-agent")
    parser.add_argument("--role-arn", required=True)
    parser.add_argument("--knowledge-base-id", required=True)
    parser.add_argument("--alias-name", default="demo")
    parser.add_argument("--question", required=True)
    parser.add_argument("--region", default=AWS_REGION)
    args = parser.parse_args()

    deployer = BedrockAgentDeployer(region=args.region)

    agent_config = AgentConfig(
        agent_name=args.agent_name,
        agent_role_arn=args.role_arn,
        knowledge_base_id=args.knowledge_base_id,
    )

    agent_id = deployer.create_or_update_agent(agent_config)
    deployer.associate_knowledge_base(agent_id, args.knowledge_base_id)
    deployer.prepare_agent(agent_id)
    alias_id = deployer.create_agent_alias(agent_id, args.alias_name)

    print(f"\nAgent ID : {agent_id}")
    print(f"Alias ID : {alias_id}")
    print(f"Question : {args.question}\n")

    runtime = boto3.client("bedrock-agent-runtime", region_name=args.region)
    response = runtime.invoke_agent(
        agentId=agent_id,
        agentAliasId=alias_id,
        sessionId=str(uuid.uuid4()),
        inputText=args.question,
    )
    answer = "".join(
        event["chunk"]["bytes"].decode("utf-8")
        for event in response.get("completion", [])
        if "chunk" in event
    )
    print(f"Answer:\n{answer}")


if __name__ == "__main__":
    main()
