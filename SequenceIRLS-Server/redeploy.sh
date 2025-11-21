#!/bin/bash

# --------------------------
# Configuration
# --------------------------
PROJECT_ID="personal-website-478523"
REGION="us-east1"
REPO="server-images"
SERVICE="serverirls"
IMAGE_NAME="$SERVICE:latest"
ARTIFACT_IMAGE="us-east1-docker.pkg.dev/$PROJECT_ID/$REPO/$IMAGE_NAME"
DOCKERFILE_PATH="."  # Adjust if Dockerfile is elsewhere

# --------------------------
# 1. Build and push Docker image
# --------------------------
echo "Building Docker image for linux/amd64 and pushing to Artifact Registry..."
docker buildx build \
  --platform linux/amd64 \
  -t $ARTIFACT_IMAGE \
  --push \
  $DOCKERFILE_PATH

if [ $? -ne 0 ]; then
    echo "Docker build/push failed. Exiting."
    exit 1
fi

# --------------------------
# 2. Deploy to Cloud Run
# --------------------------
echo "Deploying to Cloud Run..."
gcloud run deploy $SERVICE \
  --image $ARTIFACT_IMAGE \
  --region $REGION \
  --platform managed \
  --allow-unauthenticated

if [ $? -ne 0 ]; then
    echo "Cloud Run deployment failed. Exiting."
    exit 1
fi

echo "Deployment complete! Service URL:"
gcloud run services describe $SERVICE --region $REGION --format="value(status.url)"
