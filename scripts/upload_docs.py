#!/usr/bin/env python3
"""Upload local markdown documents to S3 for RAG ingestion."""

import argparse
import logging
import os
import sys
from pathlib import Path

import boto3
from botocore.exceptions import ClientError

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def upload_directory(
    local_dir: Path,
    bucket: str,
    prefix: str,
    region: str,
    dry_run: bool = False,
    extensions: list[str] | None = None,
) -> tuple[int, int]:
    extensions = extensions or [".md", ".txt"]
    s3 = boto3.client("s3", region_name=region)
    uploaded = failed = 0

    files = [f for f in local_dir.rglob("*") if f.is_file() and f.suffix in extensions]
    logger.info("Found %d files to upload from %s", len(files), local_dir)

    for file_path in sorted(files):
        relative = file_path.relative_to(local_dir)
        s3_key = f"{prefix.rstrip('/')}/{relative}".lstrip("/")

        if dry_run:
            logger.info("[DRY RUN] Would upload: %s → s3://%s/%s", file_path, bucket, s3_key)
            uploaded += 1
            continue

        try:
            s3.upload_file(
                str(file_path),
                bucket,
                s3_key,
                ExtraArgs={"ContentType": "text/markdown"},
            )
            logger.info("Uploaded: %s → s3://%s/%s", file_path.name, bucket, s3_key)
            uploaded += 1
        except ClientError as e:
            logger.error("Failed: %s — %s", file_path, e)
            failed += 1

    return uploaded, failed


def main() -> int:
    parser = argparse.ArgumentParser(description="Upload markdown docs to S3")
    parser.add_argument("--bucket", required=True, help="S3 bucket name")
    parser.add_argument("--prefix", default="docs", help="S3 key prefix (default: docs)")
    parser.add_argument("--local-dir", default="docs", help="Local directory to upload")
    parser.add_argument("--region", default="us-east-1", help="AWS region")
    parser.add_argument("--dry-run", action="store_true", help="Print actions without uploading")
    args = parser.parse_args()

    local_dir = Path(args.local_dir)
    if not local_dir.exists():
        logger.error("Directory not found: %s", local_dir)
        return 1

    uploaded, failed = upload_directory(
        local_dir=local_dir,
        bucket=args.bucket,
        prefix=args.prefix,
        region=args.region,
        dry_run=args.dry_run,
    )

    logger.info("Done. Uploaded: %d  Failed: %d", uploaded, failed)
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
