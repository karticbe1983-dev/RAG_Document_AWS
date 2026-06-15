"""Bedrock Agent deployer: create, update, and publish versioned agent aliases."""

import logging
import time
from dataclasses import dataclass, field

import boto3
from botocore.exceptions import ClientError

from config.settings import (
    AWS_REGION,
    LLM_MODEL_ID,
    AGENT_IDLE_SESSION_TTL,
    AGENT_PREPARE_TIMEOUT,
    AGENT_POLL_INTERVAL,
)

logger = logging.getLogger(__name__)


@dataclass
class AgentConfig:
    """Parameters needed to create or update a Bedrock Agent."""

    agent_name: str = "rag-document-agent"
    agent_description: str = "RAG agent for answering questions from a document knowledge base"
    foundation_model: str = LLM_MODEL_ID
    agent_role_arn: str = ""
    knowledge_base_id: str = ""
    idle_session_ttl: int = AGENT_IDLE_SESSION_TTL
    instruction: str = (
        "You are a helpful assistant that answers questions based on the provided knowledge base. "
        "Always ground your answers in the retrieved documents and cite your sources. "
        "If you cannot find relevant information, say so clearly."
    )
    tags: dict[str, str] = field(default_factory=lambda: {"Project": "RAG-Document-AWS"})


class BedrockAgentDeployer:
    """Create, update, and publish versioned aliases for an Amazon Bedrock Agent.

    Responsibility: manage the Bedrock Agent control-plane lifecycle —
    create/update, prepare (compile), associate a Knowledge Base, and publish
    an alias.  Agent invocation (runtime) is handled in ``demo/invoke_agent.py``.
    """

    def __init__(self, region: str = AWS_REGION) -> None:
        """Initialise the deployer.

        Args:
            region: AWS region where the Bedrock Agent lives.
        """
        self.region = region
        self._client = boto3.client("bedrock-agent", region_name=region)

    def create_or_update_agent(self, config: AgentConfig) -> str:
        """Create the agent if it does not exist; otherwise update it in place.

        Args:
            config: Agent configuration parameters.

        Returns:
            Bedrock Agent ID (str).
        """
        existing_id = self._find_agent(config.agent_name)
        if existing_id:
            logger.info("Updating existing agent: %s", existing_id)
            self._update_agent(existing_id, config)
            return existing_id

        logger.info("Creating new Bedrock agent: %s", config.agent_name)
        return self._create_agent(config)

    def prepare_agent(self, agent_id: str) -> bool:
        """Compile the agent so it can accept live traffic.

        Args:
            agent_id: ID of the agent to prepare.

        Returns:
            ``True`` on success; ``False`` if an error occurred.
        """
        try:
            self._client.prepare_agent(agentId=agent_id)
            self._wait_for_status(agent_id, target_status="PREPARED")
            logger.info("Agent %s prepared successfully", agent_id)
            return True
        except ClientError as e:
            logger.error("Failed to prepare agent %s: %s", agent_id, e)
            return False

    def create_agent_alias(self, agent_id: str, alias_name: str) -> str:
        """Publish a named alias pointing to the latest prepared agent version.

        Args:
            agent_id: ID of the agent to alias.
            alias_name: Human-readable alias name (e.g. ``"prod"`` or ``"dev"``).

        Returns:
            The newly created alias ID.

        Raises:
            ClientError: If the Bedrock API call fails.
        """
        try:
            response = self._client.create_agent_alias(
                agentId=agent_id,
                agentAliasName=alias_name,
                description=f"Alias '{alias_name}' for agent {agent_id}",
            )
            alias_id: str = response["agentAlias"]["agentAliasId"]
            logger.info("Created alias '%s' (%s) for agent %s", alias_name, alias_id, agent_id)
            return alias_id
        except ClientError as e:
            logger.error("Failed to create alias: %s", e)
            raise

    def associate_knowledge_base(
        self,
        agent_id: str,
        knowledge_base_id: str,
        description: str = "Primary knowledge base",
    ) -> bool:
        """Associate a Bedrock Knowledge Base with the DRAFT version of the agent.

        Args:
            agent_id: ID of the agent to update.
            knowledge_base_id: ID of the Knowledge Base to attach.
            description: Human-readable description of the association.

        Returns:
            ``True`` on success; ``False`` if an error occurred.
        """
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

    # ── Private helpers ───────────────────────────────────────────────────────

    def _create_agent(self, config: AgentConfig) -> str:
        """Call the Bedrock API to create a new agent and wait for it to initialise.

        Args:
            config: Agent creation parameters.

        Returns:
            ID of the newly created agent.
        """
        response = self._client.create_agent(
            agentName=config.agent_name,
            description=config.agent_description,
            foundationModel=config.foundation_model,
            agentResourceRoleArn=config.agent_role_arn,
            idleSessionTTLInSeconds=config.idle_session_ttl,
            instruction=config.instruction,
            tags=config.tags,
        )
        agent_id: str = response["agent"]["agentId"]
        self._wait_for_status(agent_id, target_status="NOT_PREPARED")
        return agent_id

    def _update_agent(self, agent_id: str, config: AgentConfig) -> None:
        """Push updated configuration to an existing agent.

        Args:
            agent_id: ID of the agent to update.
            config: New configuration to apply.
        """
        self._client.update_agent(
            agentId=agent_id,
            agentName=config.agent_name,
            foundationModel=config.foundation_model,
            agentResourceRoleArn=config.agent_role_arn,
            idleSessionTTLInSeconds=config.idle_session_ttl,
            instruction=config.instruction,
        )

    def _find_agent(self, agent_name: str) -> str | None:
        """Search all agents for one whose name matches *agent_name*.

        Args:
            agent_name: Display name of the agent to look for.

        Returns:
            Agent ID if found; ``None`` otherwise.
        """
        paginator = self._client.get_paginator("list_agents")
        for page in paginator.paginate():
            for agent in page.get("agentSummaries", []):
                if agent["agentName"] == agent_name:
                    return agent["agentId"]
        return None

    def _wait_for_status(
        self, agent_id: str, target_status: str
    ) -> None:
        """Poll the agent until it reaches *target_status* or timeout.

        Args:
            agent_id: ID of the agent to poll.
            target_status: Expected final status string (e.g. ``"PREPARED"``).

        Raises:
            RuntimeError: If the agent enters any FAILED state.
            TimeoutError: If *target_status* is not reached within
                ``AGENT_PREPARE_TIMEOUT`` seconds.
        """
        waited = 0
        while waited < AGENT_PREPARE_TIMEOUT:
            response = self._client.get_agent(agentId=agent_id)
            status: str = response["agent"]["agentStatus"]
            if status == target_status:
                return
            if "FAILED" in status:
                raise RuntimeError(f"Agent entered failed state: {status}")
            time.sleep(AGENT_POLL_INTERVAL)
            waited += AGENT_POLL_INTERVAL
        raise TimeoutError(
            f"Agent did not reach '{target_status}' within {AGENT_PREPARE_TIMEOUT}s"
        )
