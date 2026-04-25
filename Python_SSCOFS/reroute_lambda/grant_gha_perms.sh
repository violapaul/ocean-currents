#!/usr/bin/env bash
# One-shot: extend the github-actions-ocean-currents IAM user so the
# `Deploy Reroute Lambda` workflow can push to ECR and update the
# Lambda function. Idempotent — re-running just overwrites the inline
# policy. Read-only on existing policies; safe to run more than once.
#
# This grants ONLY:
#   - ecr:GetAuthorizationToken         (account-wide; required by the API)
#   - ecr:* (image push verbs)          on race-reroute repo only
#   - lambda:UpdateFunctionCode +
#     lambda:GetFunction                on race-reroute fn only
#
# It does not touch existing S3 perms.

set -euo pipefail

REGION="us-west-2"
USER_NAME="github-actions-ocean-currents"
POLICY_NAME="race-reroute-deploy"

ACCOUNT_ID="$(aws sts get-caller-identity --query Account --output text)"

cat > /tmp/race-reroute-deploy-policy.json <<EOF
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "ECRAuthAccountWide",
      "Effect": "Allow",
      "Action": ["ecr:GetAuthorizationToken"],
      "Resource": "*"
    },
    {
      "Sid": "ECRPushRaceReroute",
      "Effect": "Allow",
      "Action": [
        "ecr:BatchGetImage",
        "ecr:BatchCheckLayerAvailability",
        "ecr:CompleteLayerUpload",
        "ecr:InitiateLayerUpload",
        "ecr:PutImage",
        "ecr:UploadLayerPart"
      ],
      "Resource": "arn:aws:ecr:${REGION}:${ACCOUNT_ID}:repository/race-reroute"
    },
    {
      "Sid": "LambdaUpdateRaceReroute",
      "Effect": "Allow",
      "Action": [
        "lambda:GetFunction",
        "lambda:UpdateFunctionCode"
      ],
      "Resource": "arn:aws:lambda:${REGION}:${ACCOUNT_ID}:function:race-reroute"
    }
  ]
}
EOF

echo "==> Attaching inline policy ${POLICY_NAME} to user ${USER_NAME}"
aws iam put-user-policy \
  --user-name "$USER_NAME" \
  --policy-name "$POLICY_NAME" \
  --policy-document file:///tmp/race-reroute-deploy-policy.json

echo "Done."
