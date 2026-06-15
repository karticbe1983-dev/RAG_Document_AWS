# Amazon OpenSearch Serverless — vector store for RAG embeddings

resource "aws_opensearchserverless_security_policy" "encryption" {
  name        = "${var.project_name}-enc"
  type        = "encryption"
  description = "Encryption policy for RAG vector collection"
  policy = jsonencode({
    Rules = [
      {
        ResourceType = "collection"
        Resource     = ["collection/${var.opensearch_collection_name}"]
      }
    ]
    AWSOwnedKey = true
  })
}

resource "aws_opensearchserverless_security_policy" "network" {
  name        = "${var.project_name}-net"
  type        = "network"
  description = "Public access for RAG vector collection (restrict in prod)"
  policy = jsonencode([
    {
      Rules = [
        {
          ResourceType = "collection"
          Resource     = ["collection/${var.opensearch_collection_name}"]
        },
        {
          ResourceType = "dashboard"
          Resource     = ["collection/${var.opensearch_collection_name}"]
        }
      ]
      AllowFromPublic = true
    }
  ])
}

resource "aws_opensearchserverless_access_policy" "vectors" {
  name        = "${var.project_name}-access"
  type        = "data"
  description = "Data access policy for Bedrock KB and application"
  policy = jsonencode([
    {
      Rules = [
        {
          ResourceType = "index"
          Resource     = ["index/${var.opensearch_collection_name}/*"]
          Permission = [
            "aoss:CreateIndex",
            "aoss:DeleteIndex",
            "aoss:UpdateIndex",
            "aoss:DescribeIndex",
            "aoss:ReadDocument",
            "aoss:WriteDocument",
          ]
        },
        {
          ResourceType = "collection"
          Resource     = ["collection/${var.opensearch_collection_name}"]
          Permission = [
            "aoss:CreateCollectionItems",
            "aoss:DescribeCollectionItems",
            "aoss:UpdateCollectionItems",
          ]
        }
      ]
      Principal = [
        aws_iam_role.bedrock_kb.arn,
        aws_iam_role.bedrock_agent.arn,
        "arn:aws:iam::${data.aws_caller_identity.current.account_id}:root",
      ]
    }
  ])
}

resource "aws_opensearchserverless_collection" "vectors" {
  name        = var.opensearch_collection_name
  type        = "VECTORSEARCH"
  description = "Vector store for RAG document embeddings"

  depends_on = [
    aws_opensearchserverless_security_policy.encryption,
    aws_opensearchserverless_security_policy.network,
    aws_opensearchserverless_access_policy.vectors,
  ]
}
