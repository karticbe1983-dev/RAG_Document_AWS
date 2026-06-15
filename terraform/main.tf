# RAG Document AWS — root Terraform module
#
# Resources are split into focused files:
#   s3.tf        — Document and artifact S3 buckets
#   opensearch.tf — OpenSearch Serverless vector collection
#   bedrock.tf   — Bedrock Knowledge Base and Agent
#   iam.tf       — IAM roles, policies, and GitHub OIDC provider
#   outputs.tf   — Exported values used by CI/CD and the application

# Nothing to declare here — all resources live in the per-service files above.
# This file exists as the conventional entry point for `terraform plan/apply`.
