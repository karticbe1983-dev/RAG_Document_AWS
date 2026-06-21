data "aws_iam_policy_document" "bedrock_agent_assume_role" {
  statement {
    actions = ["sts:AssumeRole"]
    principals {
      type        = "Service"
      identifiers = ["bedrock.amazonaws.com"]
    }
    condition {
      test     = "StringEquals"
      variable = "aws:SourceAccount"
      values   = [data.aws_caller_identity.current.account_id]
    }
  }
}

# ── Bedrock Agent execution role ───────────────────────────────────────────

resource "aws_iam_role" "bedrock_agent" {
  name               = "${var.project_name}-bedrock-agent-role"
  assume_role_policy = data.aws_iam_policy_document.bedrock_agent_assume_role.json
}

resource "aws_iam_role_policy" "bedrock_agent_policy" {
  name   = "${var.project_name}-bedrock-agent-policy"
  role   = aws_iam_role.bedrock_agent.id
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid    = "BedrockFoundationModelAccess"
        Effect = "Allow"
        Action = [
          "bedrock:InvokeModel",
          "bedrock:InvokeModelWithResponseStream",
        ]
        Resource = [
          "arn:aws:bedrock:${var.aws_region}::foundation-model/${var.bedrock_llm_model_id}",
          "arn:aws:bedrock:${var.aws_region}::foundation-model/${var.bedrock_embed_model_id}",
        ]
      },
      {
        Sid    = "KnowledgeBaseAccess"
        Effect = "Allow"
        Action = [
          "bedrock:Retrieve",
          "bedrock:RetrieveAndGenerate",
        ]
        Resource = aws_bedrockagent_knowledge_base.main.arn
      },
      {
        Sid      = "S3DocumentsRead"
        Effect   = "Allow"
        Action   = ["s3:GetObject", "s3:ListBucket"]
        Resource = [
          aws_s3_bucket.documents.arn,
          "${aws_s3_bucket.documents.arn}/*",
        ]
      },
    ]
  })
}

# ── Bedrock Knowledge Base role ────────────────────────────────────────────

resource "aws_iam_role" "bedrock_kb" {
  name               = "${var.project_name}-bedrock-kb-role"
  assume_role_policy = data.aws_iam_policy_document.bedrock_agent_assume_role.json
}

resource "aws_iam_role_policy" "bedrock_kb_policy" {
  name   = "${var.project_name}-bedrock-kb-policy"
  role   = aws_iam_role.bedrock_kb.id
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid    = "EmbeddingModelAccess"
        Effect = "Allow"
        Action = ["bedrock:InvokeModel"]
        Resource = [
          "arn:aws:bedrock:${var.aws_region}::foundation-model/${var.bedrock_embed_model_id}",
        ]
      },
      {
        Sid      = "S3DataSourceRead"
        Effect   = "Allow"
        Action   = ["s3:GetObject", "s3:ListBucket"]
        Resource = [
          aws_s3_bucket.documents.arn,
          "${aws_s3_bucket.documents.arn}/*",
        ]
      },
      {
        Sid    = "OpenSearchAccess"
        Effect = "Allow"
        Action = ["aoss:APIAccessAll"]
        Resource = aws_opensearchserverless_collection.vectors.arn
      },
    ]
  })
}

# ── CI/CD deployment role (assumed by GitHub Actions via OIDC) ─────────────

resource "aws_iam_openid_connect_provider" "github" {
  count = 1

  url = "https://token.actions.githubusercontent.com"
  client_id_list = ["sts.amazonaws.com"]
  thumbprint_list = [
    "6938fd4d98bab03faadb97b34396831e3780aea1",
    "1c58a3a8518e8759bf075b76b750d4f2df264fcd",
  ]
}

resource "aws_iam_role" "github_actions" {
  name = "${var.project_name}-github-actions-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect = "Allow"
      Principal = {
        Federated = aws_iam_openid_connect_provider.github[0].arn
      }
      Action = "sts:AssumeRoleWithWebIdentity"
      Condition = {
        StringLike = {
          "token.actions.githubusercontent.com:sub" = "repo:karticbe1983-dev/RAG_Document_AWS:ref:refs/heads/main"
        }
        StringEquals = {
          "token.actions.githubusercontent.com:aud" = "sts.amazonaws.com"
        }
      }
    }]
  })
}

resource "aws_iam_role_policy" "github_actions_policy" {
  name   = "${var.project_name}-github-actions-policy"
  role   = aws_iam_role.github_actions.id
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid    = "TerraformStateAccess"
        Effect = "Allow"
        Action = ["s3:GetObject", "s3:PutObject", "s3:DeleteObject", "s3:ListBucket",
                  "s3:GetBucketVersioning", "s3:GetBucketAcl", "s3:GetBucketLocation"]
        Resource = ["arn:aws:s3:::*-tfstate*", "arn:aws:s3:::*-tfstate*/*"]
      },
      {
        Sid    = "S3Manage"
        Effect = "Allow"
        Action = [
          "s3:CreateBucket", "s3:DeleteBucket",
          "s3:GetObject", "s3:PutObject", "s3:DeleteObject", "s3:ListBucket",
          "s3:GetBucketPolicy", "s3:PutBucketPolicy", "s3:DeleteBucketPolicy",
          "s3:GetBucketVersioning", "s3:PutBucketVersioning",
          "s3:GetEncryptionConfiguration", "s3:PutEncryptionConfiguration",
          "s3:GetLifecycleConfiguration", "s3:PutLifecycleConfiguration",
          "s3:GetBucketPublicAccessBlock", "s3:PutBucketPublicAccessBlock",
          "s3:GetBucketAcl", "s3:GetBucketLocation", "s3:GetBucketCORS",
          "s3:ListAllMyBuckets",
        ]
        Resource = "*"
      },
      {
        Sid    = "OpenSearchServerlessManage"
        Effect = "Allow"
        Action = [
          "aoss:CreateCollection", "aoss:DeleteCollection", "aoss:UpdateCollection",
          "aoss:BatchGetCollection", "aoss:ListCollections",
          "aoss:CreateSecurityPolicy", "aoss:DeleteSecurityPolicy",
          "aoss:UpdateSecurityPolicy", "aoss:GetSecurityPolicy", "aoss:ListSecurityPolicies",
          "aoss:CreateAccessPolicy", "aoss:DeleteAccessPolicy",
          "aoss:UpdateAccessPolicy", "aoss:GetAccessPolicy", "aoss:ListAccessPolicies",
          "aoss:APIAccessAll",
        ]
        Resource = "*"
      },
      {
        Sid    = "BedrockManage"
        Effect = "Allow"
        Action = [
          "bedrock:CreateKnowledgeBase", "bedrock:DeleteKnowledgeBase",
          "bedrock:UpdateKnowledgeBase", "bedrock:GetKnowledgeBase", "bedrock:ListKnowledgeBases",
          "bedrock:CreateDataSource", "bedrock:DeleteDataSource",
          "bedrock:UpdateDataSource", "bedrock:GetDataSource",
          "bedrock:StartIngestionJob", "bedrock:GetIngestionJob", "bedrock:ListIngestionJobs",
          "bedrock:CreateAgent", "bedrock:DeleteAgent", "bedrock:UpdateAgent",
          "bedrock:GetAgent", "bedrock:ListAgents", "bedrock:PrepareAgent",
          "bedrock:CreateAgentAlias", "bedrock:DeleteAgentAlias",
          "bedrock:UpdateAgentAlias", "bedrock:GetAgentAlias", "bedrock:ListAgentAliases",
          "bedrock:AssociateAgentKnowledgeBase", "bedrock:DisassociateAgentKnowledgeBase",
          "bedrock:GetAgentKnowledgeBase", "bedrock:ListAgentKnowledgeBases",
          "bedrock:TagResource", "bedrock:UntagResource", "bedrock:ListTagsForResource",
        ]
        Resource = "*"
      },
      {
        Sid    = "IAMManage"
        Effect = "Allow"
        Action = [
          "iam:CreateRole", "iam:DeleteRole", "iam:UpdateRole", "iam:GetRole",
          "iam:PutRolePolicy", "iam:DeleteRolePolicy", "iam:GetRolePolicy",
          "iam:AttachRolePolicy", "iam:DetachRolePolicy", "iam:ListAttachedRolePolicies",
          "iam:ListRolePolicies", "iam:PassRole",
          "iam:TagRole", "iam:UntagRole", "iam:ListRoleTags",
          "iam:CreateOpenIDConnectProvider", "iam:DeleteOpenIDConnectProvider",
          "iam:GetOpenIDConnectProvider", "iam:UpdateOpenIDConnectProvider",
          "iam:ListOpenIDConnectProviders", "iam:TagOpenIDConnectProvider",
        ]
        Resource = "*"
      },
      {
        Sid    = "CloudWatchLogsManage"
        Effect = "Allow"
        Action = [
          "logs:CreateLogGroup", "logs:DeleteLogGroup", "logs:PutRetentionPolicy",
          "logs:DescribeLogGroups", "logs:ListTagsLogGroup",
          "logs:TagLogGroup", "logs:UntagLogGroup", "logs:ListTagsForResource",
          "logs:TagResource",
        ]
        Resource = "*"
      },
      {
        Sid      = "STSCallerIdentity"
        Effect   = "Allow"
        Action   = ["sts:GetCallerIdentity"]
        Resource = "*"
      },
    ]
  })
}
