#!/bin/bash

# Deploy to Google Cloud Run
# Usage: ./deploy.sh [SERVICE_NAME] [REGION]

SERVICE_NAME=${1:-bluesky-feed-server}
REGION=${2:-us-central1}

echo "Deploying Bluesky Feed Server to Google Cloud Run..."
echo "Service: $SERVICE_NAME"
echo "Region: $REGION"

# Deploy to Cloud Run
gcloud run deploy $SERVICE_NAME \
    --source . \
    --platform managed \
    --region $REGION \
    --allow-unauthenticated \
    --port 8080 \
    --memory 1Gi \
    --cpu 1 \
    --min-instances 0 \
    --max-instances 10 \
    --timeout 60 \
    --set-env-vars "PORT=8080,HOST=0.0.0.0" \
    --set-env-vars "REDIS_URL=$REDIS_URL" \
    --set-env-vars "FEED_DID=$FEED_DID" \
    --set-env-vars "FEED_URI=$FEED_URI" \
    --set-env-vars "FEED_CID=$FEED_CID"

echo "Deployment complete!"
echo "Your feed server URL will be displayed above."