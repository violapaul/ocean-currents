#!/usr/bin/env bash
# One-time AWS setup for the race-reroute Lambda.
# Idempotent: re-running is safe (each create checks for existence first).
#
# Provisions, in order:
#   1. ECR repository: race-reroute
#   2. IAM role:       race-reroute-exec  (logs + S3:GetObject scoped)
#   3. Lambda fn:      race-reroute       (after a first image push)
#   4. Function URL:   public, CORS allowed
#
# Run AFTER ./deploy.sh push has put a first image in ECR. Subsequent
# code updates use ./deploy.sh deploy (no need to re-run setup).

set -euo pipefail

REGION="us-west-2"
REPO="race-reroute"
ROLE_NAME="race-reroute-exec"
LAMBDA_FN="race-reroute"
IMAGE_TAG="latest"

ACCOUNT_ID="$(aws sts get-caller-identity --query Account --output text)"
ECR_URI="${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com/${REPO}"

echo "==> Account: $ACCOUNT_ID  region: $REGION"

# --- 1. ECR repository ---------------------------------------------------
if ! aws ecr describe-repositories --repository-names "$REPO" --region "$REGION" >/dev/null 2>&1; then
  echo "==> Creating ECR repository $REPO"
  aws ecr create-repository \
    --repository-name "$REPO" \
    --region "$REGION" \
    --image-scanning-configuration scanOnPush=true >/dev/null
else
  echo "==> ECR repository $REPO exists"
fi

# --- 2. IAM execution role ----------------------------------------------
TRUST='{"Version":"2012-10-17","Statement":[{"Effect":"Allow","Principal":{"Service":"lambda.amazonaws.com"},"Action":"sts:AssumeRole"}]}'

if ! aws iam get-role --role-name "$ROLE_NAME" >/dev/null 2>&1; then
  echo "==> Creating IAM role $ROLE_NAME"
  aws iam create-role \
    --role-name "$ROLE_NAME" \
    --assume-role-policy-document "$TRUST" >/dev/null
  # Basic CloudWatch logs perms
  aws iam attach-role-policy \
    --role-name "$ROLE_NAME" \
    --policy-arn arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole >/dev/null
else
  echo "==> IAM role $ROLE_NAME exists"
fi

# Inline policy: read-only access to the SSCOFS cache + ECMWF / NOAA upstreams
# go over plain HTTPS, no AWS credentials needed. We don't grant any S3
# write perms — reroute returns JSON inline; nothing is published.
cat > /tmp/race-reroute-policy.json <<'EOF'
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "ReadOceanCurrentsBucket",
      "Effect": "Allow",
      "Action": ["s3:GetObject", "s3:ListBucket"],
      "Resource": [
        "arn:aws:s3:::viola-ocean-currents",
        "arn:aws:s3:::viola-ocean-currents/*"
      ]
    }
  ]
}
EOF

aws iam put-role-policy \
  --role-name "$ROLE_NAME" \
  --policy-name "race-reroute-s3-read" \
  --policy-document file:///tmp/race-reroute-policy.json >/dev/null
echo "==> Inline policy race-reroute-s3-read attached"

ROLE_ARN="$(aws iam get-role --role-name "$ROLE_NAME" --query 'Role.Arn' --output text)"
echo "==> Role ARN: $ROLE_ARN"

# Wait briefly for IAM eventual consistency (otherwise create-function may
# fail on first run with "role cannot be assumed").
sleep 8

# --- 3. Lambda function -------------------------------------------------
if aws lambda get-function --function-name "$LAMBDA_FN" --region "$REGION" >/dev/null 2>&1; then
  echo "==> Lambda function $LAMBDA_FN exists (use ./deploy.sh deploy to update)"
elif aws ecr describe-images --repository-name "$REPO" --image-ids imageTag="$IMAGE_TAG" --region "$REGION" >/dev/null 2>&1; then
  echo "==> Creating Lambda function $LAMBDA_FN"
  aws lambda create-function \
    --function-name "$LAMBDA_FN" \
    --package-type Image \
    --code "ImageUri=${ECR_URI}:${IMAGE_TAG}" \
    --role "$ROLE_ARN" \
    --architectures x86_64 \
    --memory-size 3008 \
    --timeout 90 \
    --region "$REGION" >/dev/null
else
  echo "==> Lambda function $LAMBDA_FN cannot be created yet — no image in ECR."
  echo "    Push an image (./deploy.sh push, or trigger Deploy Reroute Lambda)"
  echo "    workflow), then re-run this script. ECR + IAM are already provisioned."
  echo
  echo "==> setup.sh: ECR + IAM done, Lambda + Function URL pending."
  exit 0
fi

# --- 4. Function URL ----------------------------------------------------
if ! aws lambda get-function-url-config --function-name "$LAMBDA_FN" --region "$REGION" >/dev/null 2>&1; then
  echo "==> Creating Function URL"
  aws lambda create-function-url-config \
    --function-name "$LAMBDA_FN" \
    --auth-type NONE \
    --cors '{"AllowOrigins":["*"],"AllowMethods":["POST"],"AllowHeaders":["Content-Type"],"MaxAge":300}' \
    --region "$REGION" >/dev/null
  # Public URL must also have an explicit invoke permission for principal *.
  aws lambda add-permission \
    --function-name "$LAMBDA_FN" \
    --statement-id PublicFunctionURL \
    --action lambda:InvokeFunctionUrl \
    --principal '*' \
    --function-url-auth-type NONE \
    --region "$REGION" >/dev/null
else
  echo "==> Function URL exists"
fi

URL="$(aws lambda get-function-url-config --function-name "$LAMBDA_FN" --region "$REGION" --query 'FunctionUrl' --output text)"
echo
echo "==> DONE. Function URL:"
echo "    $URL"
echo
echo "Smoke test:"
echo "  curl -sf -X POST '$URL' \\"
echo "    -H 'Content-Type: application/json' \\"
echo "    -d @OceanCurrents/Python_SSCOFS/reroute_lambda/test_event.json"
