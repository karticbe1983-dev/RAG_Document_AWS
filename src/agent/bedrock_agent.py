import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any
import boto3
from botocore.exceptions import ClientError

logger = logging.getLogger(__name__)


@dataclass
class AgentConfig:
    agent_name: str = "rag-document-agent"
    agent_description: str = "RAG agent for answering questions from document knowledge base"
    foundation_model: str = "anthropic.claude-3-5-sonnet-20241022-v2:0"
    agent_role_arn: str = ""
    knowledge_base_id: str = ""
    idle_session_ttl: int = 600
    instruction: str = (
        "You are a helpful assistant that answers questions based on the provided knowledge base. "
        "Always ground your answers in the retrieved documents and cite your sources. "
        "If you cannot find relevant information, say so clearly."
    )
    tags: dict[str, str] = field(default_factory=lambda: {"Project": "RAG-Document-AWS"})


class BedrockAgentDeployer:
    """Create, update, and manage an Amazon Bedrock Agent with a Knowledge Base."""

    def __init__(self, region: str = "us-east-1"):
        self.region = region
        self._client = boto3.client("bedrock-agent", region_name=region)

    def create_or_update_agent(self, config: AgentConfig) -> str:
        """Create agent if it doesn't exist, otherwise update it. Returns agent_id."""
        existing_id = self._find_agent(config.agent_name)
        if existing_id:
            logger.info("Updating existing agent: %s", existing_id)
            self._update_agent(existing_id, config)
            return existing_id

        logger.info("Creating new Bedrock agent: %s", config.agent_name)
        return self._create_agent(config)

    def prepare_agent(self, agent_id: str) -> bool:
        """Prepare (compile) the agent so it can be tested."""
        try:
            self._client.prepare_agent(agentId=agent_id)
            self._wait_for_agent_status(agent_id, target_status="PREPARED")
            logger.info("Agent %s prepared successfully", agent_id)
            return True
        except ClientError as e:
            logger.error("Failed to prepare agent %s: %s", agent_id, e)
            return False

    def create_agent_alias(self, agent_id: str, alias_name: str = "prod") -> str:
        """Create an agent alias pointing to the latest prepared version."""
        try:
            response = self._client.create_agent_alias(
                agentId=agent_id,
                agentAliasName=alias_name,
                description=f"Production alias for {alias_name}",
            )
            alias_id = response["agentAlias"]["agentAliasId"]
            logger.info("Created alias '%s' (%s) for agent %s", alias_name, alias_id, agent_id)
            return alias_id
        except ClientError as e:
            logger.error("Failed to create alias: %s", e)
            raise

    def associate_knowledge_base(
        self, agent_id: str, knowledge_base_id: str, description: str = "Primary knowledge base"
    ) -> bool:
        try:
            self._client.associate_agent_knowledge_base(
                agentId=agent_id,
                agentVersion="DRAFT",
                knowledgeBaseId=knowledge_base_id,
                description=description,
                knowledgeBaseState="ENABLED",
            )
            logger.info("Associated KB %s with agent %s", knowledge_base_id, agent_id)
            return True
        except ClientError as e:
            logger.error("KB association failed: %s", e)
            return False

    def delete_agent(self, agent_id: str) -> bool:
        try:
            self._client.delete_agent(agentId=agent_id, skipResourceInUseCheck=True)
            logger.info("Deleted agent %s", agent_id)
            return True
        except ClientError as e:
            logger.error("Failed to delete agent %s: %s", agent_id, e)
            return False

    def invoke_agent(
        self, agent_id: str, alias_id: str, session_id: str, prompt: str
    ) -> dict[str, Any]:
        """Send a message to a deployed agent and return the response."""
        runtime = boto3.client("bedrock-agent-runtime", region_name=self.region)
        try:
            response = runtime.invoke_agent(
                agentId=agent_id,
                agentAliasId=alias_id,
                sessionId=session_id,
                inputText=prompt,
            )
            completion = ""
            for event in response.get("completion", []):
                if "chunk" in event:
                    completion += event["chunk"]["bytes"].decode("utf-8")
            return {"answer": completion, "session_id": session_id}
        except ClientError as e:
            logger.error("Agent invocation error: %s", e)
            raise

    # ──────────────────────────────────────────────────────────────────────

    def _create_agent(self, config: AgentConfig) -> str:
        response = self._client.create_agent(
            agentName=config.agent_name,
            description=config.description if hasattr(config, "description") else config.agent_description,
            foundationModel=config.foundation_model,
            agentResourceRoleArn=config.agent_role_arn,
            idleSessionTTLInSeconds=config.idle_session_ttl,
            instruction=config.instruction,
            tags=config.tags,
        )
        agent_id = response["agent"]["agentId"]
        self._wait_for_agent_status(agent_id, target_status="NOT_PREPARED")
        return agent_id

    def _update_agent(self, agent_id: str, config: AgentConfig) -> None:
        self._client.update_agent(
            agentId=agent_id,
            agentName=config.agent_name,
            foundationModel=config.foundation_model,
            agentResourceRoleArn=config.agent_role_arn,
            idleSessionTTLInSeconds=config.idle_session_ttl,
            instruction=config.instruction,
        )

    def _find_agent(self, agent_name: str) -> str | None:
        paginator = self._client.get_paginator("list_agents")
        for page in paginator.paginate():
            for agent in page.get("agentSummaries", []):
                if agent["agentName"] == agent_name:
                    return agent["agentId"]
        return None

    def _wait_for_agent_status(
        self, agent_id: str, target_status: str, max_wait: int = 120
    ) -> None:
        waited = 0
        while waited < max_wait:
            response = self._client.get_agent(agentId=agent_id)
            status = response["agent"]["agentStatus"]
            if status == target_status:
                return
            if "FAILED" in status:
                raise RuntimeError(f"Agent entered failed state: {status}")
            time.sleep(5)
            waited += 5
        raise TimeoutError(f"Agent did not reach '{target_status}' within {max_wait}s")
