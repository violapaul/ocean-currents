#!/usr/bin/env bash
# Build + push the race-reroute Lambda container image.
#
# First-time use: see README.md for the one-time ECR repo / IAM role /
# Lambda function setup. After that's done, re-running this script is
# the entire deploy loop.
#
# Usage:
#   ./deploy.sh build           # local build only (no AWS)
#   ./deploy.sh push            # build + push to ECR (no Lambda update)
#   ./deploy.sh deploy          # build + push + update Lambda function code

set -euo pipefail

REGION="us-west-2"
REPO="race-reroute"
IMAGE_TAG="latest"
LAMBDA_FN="race-reroute"

# Cd to repo root (Docker build context).
REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "$REPO_ROOT"

# Resolve account id once.
ACCOUNT_ID="$(aws sts get-caller-identity --query Account --output text)"
ECR_URI="${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com/${REPO}"

mode="${1:-deploy}"

echo "==> Build context: $REPO_ROOT"
echo "==> ECR URI: ${ECR_URI}:${IMAGE_TAG}"

# 1. Build
docker build \
  --platform linux/amd64 \
  -f OceanCurrents/Python_SSCOFS/reroute_lambda/Dockerfile \
  -t "${REPO}:${IMAGE_TAG}" \
  .

if [[ "$mode" == "build" ]]; then
  echo "==> Built ${REPO}:${IMAGE_TAG}. Stopping (mode=build)."
  exit 0
fi

# 2. Auth Docker to ECR
aws ecr get-login-password --region "$REGION" \
  | docker login --username AWS --password-stdin "${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com"

# 3. Tag + push
docker tag  "${REPO}:${IMAGE_TAG}" "${ECR_URI}:${IMAGE_TAG}"
docker push "${ECR_URI}:${IMAGE_TAG}"
echo "==> Pushed ${ECR_URI}:${IMAGE_TAG}"

if [[ "$mode" == "push" ]]; then
  exit 0
fi

# 4. Update the existing Lambda function to the new image
aws lambda update-function-code \
  --function-name "$LAMBDA_FN" \
  --image-uri "${ECR_URI}:${IMAGE_TAG}" \
  --region "$REGION" \
  --output table \
  --query 'FunctionName,LastUpdateStatus,LastModified'

echo "==> Lambda code update issued. Tail with:"
echo "    aws logs tail /aws/lambda/${LAMBDA_FN} --follow --region $REGION"
