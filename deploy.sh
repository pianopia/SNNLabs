#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

PROJECT_ID="pianopia"
PRODUCT_ID="elfentier-app"
REGION="asia-northeast1"

gcloud config set project "$PROJECT_ID"

gcloud artifacts repositories create "$PRODUCT_ID" \
  --repository-format=docker \
  --location="$REGION" \
  --description="Geo3 media" 2>/dev/null || true

docker buildx build --platform linux/amd64 \
  -t "$REGION-docker.pkg.dev/$PROJECT_ID/$PRODUCT_ID/$PRODUCT_ID:latest" .

docker push "$REGION-docker.pkg.dev/$PROJECT_ID/$PRODUCT_ID/$PRODUCT_ID:latest"

gcloud run deploy "$PRODUCT_ID" \
  --image "$REGION-docker.pkg.dev/$PROJECT_ID/$PRODUCT_ID/$PRODUCT_ID:latest" \
  --platform managed \
  --region="$REGION" \
  --allow-unauthenticated

gcloud run services update-traffic "$PRODUCT_ID" --to-latest --region="$REGION"
