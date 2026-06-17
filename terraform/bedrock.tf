# Amazon Bedrock Knowledge Base + Agent

resource "aws_bedrockagent_knowledge_base" "main" {
  name        = var.knowledge_base_name
  description = "Knowledge base for RAG document system"
  role_arn    = aws_iam_role.bedrock_kb.arn

  knowledge_base_configuration {
    type = "VECTOR"
    vector_knowledge_base_configuration {
      embedding_model_arn = "arn:aws:bedrock:${var.aws_region}::foundation-model/${var.bedrock_embed_model_id}"
    }
  }

  storage_configuration {
    type = "OPENSEARCH_SERVERLESS"
    opensearch_serverless_configuration {
      collection_arn    = aws_opensearchserverless_collection.vectors.arn
      vector_index_name = "rag-index"
      field_mapping {
        vector_field   = "embedding"
        text_field     = "content"
        metadata_field = "metadata"
      }
    }
  }

  depends_on = [
    aws_iam_role_policy.bedrock_kb_policy,
    null_resource.create_vector_index,
  ]
}

resource "aws_bedrockagent_data_source" "documents" {
  knowledge_base_id = aws_bedrockagent_knowledge_base.main.id
  name              = "s3-documents"
  description       = "Markdown documents from S3"

  data_source_configuration {
    type = "S3"
    s3_configuration {
      bucket_arn = aws_s3_bucket.documents.arn
    }
  }

  vector_ingestion_configuration {
    chunking_configuration {
      chunking_strategy = "FIXED_SIZE"
      fixed_size_chunking_configuration {
        max_tokens         = 512
        overlap_percentage = 20
      }
    }
  }
}

# ── Bedrock Agent ──────────────────────────────────────────────────────────

resource "aws_bedrockagent_agent" "main" {
  agent_name              = var.agent_name
  agent_resource_role_arn = aws_iam_role.bedrock_agent.arn
  foundation_model        = var.bedrock_llm_model_id
  description             = "RAG agent for document Q&A with multiple chunking strategies"
  idle_session_ttl_in_seconds = 600

  instruction = <<-EOT
    You are a knowledgeable assistant that answers questions using the provided knowledge base.

    Guidelines:
    - Always base your answers on retrieved documents
    - Cite the source document and section for each fact
    - If the knowledge base doesn't contain relevant information, say so
    - Be concise, accurate, and helpful
    - For technical questions, include code examples where appropriate
  EOT

  prepare_agent = true

  depends_on = [
    aws_iam_role_policy.bedrock_agent_policy,
  ]
}

resource "aws_bedrockagent_agent_knowledge_base_association" "main" {
  agent_id             = aws_bedrockagent_agent.main.agent_id
  description          = "Primary knowledge base for document retrieval"
  knowledge_base_id    = aws_bedrockagent_knowledge_base.main.id
  knowledge_base_state = "ENABLED"
}

resource "aws_bedrockagent_agent_alias" "prod" {
  agent_alias_name = "prod"
  agent_id         = aws_bedrockagent_agent.main.agent_id
  description      = "Production alias"

  depends_on = [aws_bedrockagent_agent_knowledge_base_association.main]
}

# ── CloudWatch logging for Bedrock ─────────────────────────────────────────

resource "aws_cloudwatch_log_group" "bedrock_agent" {
  name              = "/aws/bedrock/agent/${var.agent_name}"
  retention_in_days = var.log_retention_days
}
