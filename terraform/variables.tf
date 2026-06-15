variable "aws_region" {
  description = "AWS region for all resources"
  type        = string
  default     = "us-east-1"
}

variable "project_name" {
  description = "Project name used for resource naming and tagging"
  type        = string
  default     = "rag-document-aws"
}

variable "environment" {
  description = "Deployment environment (dev, staging, prod)"
  type        = string
  default     = "dev"

  validation {
    condition     = contains(["dev", "staging", "prod"], var.environment)
    error_message = "environment must be dev, staging, or prod."
  }
}

variable "s3_document_bucket_name" {
  description = "Name of the S3 bucket for RAG documents. Must be globally unique."
  type        = string
  default     = ""
}

variable "s3_artifacts_bucket_name" {
  description = "Name of the S3 bucket for Bedrock artifacts. Must be globally unique."
  type        = string
  default     = ""
}

variable "opensearch_collection_name" {
  description = "Name of the OpenSearch Serverless collection for vector storage"
  type        = string
  default     = "rag-vectors"
}

variable "bedrock_embed_model_id" {
  description = "Bedrock embedding model ID"
  type        = string
  default     = "amazon.titan-embed-text-v2:0"
}

variable "bedrock_llm_model_id" {
  description = "Bedrock LLM model ID for agent and retrieval"
  type        = string
  default     = "anthropic.claude-3-5-sonnet-20241022-v2:0"
}

variable "embedding_dimensions" {
  description = "Dimensions for the embedding model"
  type        = number
  default     = 1024
}

variable "knowledge_base_name" {
  description = "Name for the Bedrock Knowledge Base"
  type        = string
  default     = "rag-document-kb"
}

variable "agent_name" {
  description = "Name for the Bedrock Agent"
  type        = string
  default     = "rag-document-agent"
}

variable "enable_s3_versioning" {
  description = "Enable versioning on the documents S3 bucket"
  type        = bool
  default     = true
}

variable "log_retention_days" {
  description = "CloudWatch log retention in days"
  type        = number
  default     = 30
}
