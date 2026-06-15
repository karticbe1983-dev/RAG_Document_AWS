output "documents_bucket_name" {
  description = "S3 bucket name for RAG documents"
  value       = aws_s3_bucket.documents.bucket
}

output "artifacts_bucket_name" {
  description = "S3 bucket name for Bedrock artifacts"
  value       = aws_s3_bucket.artifacts.bucket
}

output "opensearch_collection_endpoint" {
  description = "OpenSearch Serverless collection endpoint"
  value       = aws_opensearchserverless_collection.vectors.collection_endpoint
}

output "opensearch_collection_arn" {
  description = "OpenSearch Serverless collection ARN"
  value       = aws_opensearchserverless_collection.vectors.arn
}

output "knowledge_base_id" {
  description = "Bedrock Knowledge Base ID"
  value       = aws_bedrockagent_knowledge_base.main.id
}

output "knowledge_base_arn" {
  description = "Bedrock Knowledge Base ARN"
  value       = aws_bedrockagent_knowledge_base.main.arn
}

output "data_source_id" {
  description = "Bedrock Knowledge Base data source ID"
  value       = aws_bedrockagent_data_source.documents.data_source_id
}

output "agent_id" {
  description = "Bedrock Agent ID"
  value       = aws_bedrockagent_agent.main.agent_id
}

output "agent_arn" {
  description = "Bedrock Agent ARN"
  value       = aws_bedrockagent_agent.main.agent_arn
}

output "agent_alias_id" {
  description = "Bedrock Agent production alias ID"
  value       = aws_bedrockagent_agent_alias.prod.agent_alias_id
}

output "bedrock_agent_role_arn" {
  description = "IAM role ARN for Bedrock Agent"
  value       = aws_iam_role.bedrock_agent.arn
}

output "github_actions_role_arn" {
  description = "IAM role ARN for GitHub Actions OIDC deployment"
  value       = aws_iam_role.github_actions.arn
}
