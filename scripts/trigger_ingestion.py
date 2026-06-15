#!/usr/bin/env python3
"""Start a Bedrock Knowledge Base ingestion job and wait for it to complete."""

import argparse
import logging
import sys
import time
import uuid

import boto3
from botocore.exceptions import ClientError

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

POLL_INTERVAL = 10
MAX_WAIT_SECONDS = 600


def start_ingestion(knowledge_base_id: str, data_source_id: str, region: str) -> str:
    client = boto3.client("bedrock-agent", region_name=region)
    client_token = str(uuid.uuid4())
    try:
        response = client.start_ingestion_job(
            knowledgeBaseId=knowledge_base_id,
            dataSourceId=data_source_id,
            clientToken=client_token,
            description="Triggered by CI/CD pipeline",
        )
        job_id = response["ingestionJob"]["ingestionJobId"]
        logger.info("Started ingestion job: %s", job_id)
        return job_id
    except ClientError as e:
        logger.error("Failed to start ingestion: %s", e)
        raise


def wait_for_ingestion(
    knowledge_base_id: str, data_source_id: str, job_id: str, region: str
) -> bool:
    client = boto3.client("bedrock-agent", region_name=region)
    waited = 0

    while waited < MAX_WAIT_SECONDS:
        try:
            response = client.get_ingestion_job(
                knowledgeBaseId=knowledge_base_id,
                dataSourceId=data_source_id,
                ingestionJobId=job_id,
            )
            job = response["ingestionJob"]
            status = job["status"]
            stats = job.get("statistics", {})

            logger.info(
                "Status: %s | Scanned: %s | Indexed: %s | Failed: %s",
                status,
                stats.get("numberOfDocumentsScanned", 0),
                stats.get("numberOfNewDocumentsIndexed", 0),
                stats.get("numberOfDocumentsFailed", 0),
            )

            if status == "COMPLETE":
                logger.info("Ingestion job completed successfully")
                return True
            if status in ("FAILED", "STOPPED"):
                logger.error("Ingestion job %s: %s", status, job.get("failureReasons", []))
                return False

        except ClientError as e:
            logger.error("Error polling job: %s", e)
            return False

        time.sleep(POLL_INTERVAL)
        waited += POLL_INTERVAL

    logger.error("Ingestion job timed out after %ds", MAX_WAIT_SECONDS)
    return False


def main() -> int:
    parser = argparse.ArgumentParser(description="Trigger Bedrock KB ingestion")
    parser.add_argument("--knowledge-base-id", required=True)
    parser.add_argument("--data-source-id", required=True)
    parser.add_argument("--region", default="us-east-1")
    parser.add_argument("--no-wait", action="store_true", help="Don't wait for completion")
    args = parser.parse_args()

    job_id = start_ingestion(args.knowledge_base_id, args.data_source_id, args.region)
    if args.no_wait:
        logger.info("Job started (--no-wait): %s", job_id)
        return 0

    success = wait_for_ingestion(
        args.knowledge_base_id, args.data_source_id, job_id, args.region
    )
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
