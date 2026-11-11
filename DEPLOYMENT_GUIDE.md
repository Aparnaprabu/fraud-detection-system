# Production Deployment Guide

This guide describes deployment options for the Real-Time Fraud Detection System.

**Deployment status:** The application has been tested locally using Docker. Cloud deployment has not yet been performed.

## Architecture

- **Backend:** FastAPI fraud prediction API.
- **Frontend:** Streamlit dashboard.
- **Model:** XGBoost with SHAP explanations.
- **Containerization:** Docker and Docker Compose.

## Option 1: Google Cloud Run

Deploy the FastAPI backend as a containerized service.

### Prerequisites

- Google Cloud account.
- Google Cloud CLI installed.
- A configured Google Cloud project.
- Required Cloud Run and Artifact Registry permissions.

### Step 1: Configure Google Cloud

```bash
export PROJECT_ID="your-gcp-project-id"
export REGION="us-central1"

gcloud config set project "$PROJECT_ID"

gcloud services enable \
  run.googleapis.com \
  artifactregistry.googleapis.com \
  cloudbuild.googleapis.com
```

### Step 2: Build the container

```bash
gcloud builds submit \
  --tag "gcr.io/$PROJECT_ID/fraud-detection-api:latest" .
```

### Step 3: Deploy the API

```bash
gcloud run deploy fraud-detection-api \
  --image "gcr.io/$PROJECT_ID/fraud-detection-api:latest" \
  --region "$REGION" \
  --allow-unauthenticated \
  --port 8000 \
  --memory 1Gi \
  --cpu 1
```

After deployment, Google Cloud Run provides a service URL.

Test the health endpoint:

```bash
curl https://YOUR_SERVICE_URL/health
```

## Option 2: Render

The project includes a `render.yaml` Blueprint configuration for deploying the API and dashboard.

### Deployment steps

1. Push the project to GitHub.
2. Sign in at https://render.com/.
3. Select **New → Blueprint**.
4. Connect the `fraud-detection-system` GitHub repository.
5. Review the services and environment variables detected from `render.yaml`.
6. Deploy after confirming the Docker configuration and health checks.

The API and dashboard must use compatible ports, and the dashboard must receive the complete API base URL.

**Important:** Review the current Dockerfile and Render configuration before deployment. The deployment configuration has not yet been verified in a live Render environment.

## Option 3: Streamlit Community Cloud

Streamlit Community Cloud can host the dashboard separately from the API.

### Deployment steps

1. Deploy the FastAPI backend to a hosting provider.
2. Open https://share.streamlit.io/.
3. Sign in using GitHub.
4. Select **New app**.
5. Choose the repository: `Aparnaprabu/fraud-detection-system`.
6. Select branch: `main`.
7. Set the main file path to `dashboard/app.py`.
8. Configure the API URL in the app's secrets or environment settings:

```toml
API_URL = "https://YOUR_DEPLOYED_API_URL"
```

9. Deploy the dashboard.

## Deployment Verification Checklist

After deployment, verify:

- [ ] API health endpoint returns a successful response.
- [ ] Model loads successfully.
- [ ] Prediction endpoint returns a valid response.
- [ ] Fraud probability and decision are displayed correctly.
- [ ] SHAP explanations are available.
- [ ] Dashboard successfully connects to the API.
- [ ] Manual transaction testing works.
- [ ] API latency is monitored.

## Security Considerations

- Never commit API keys, passwords, or access tokens.
- Configure secrets using the hosting provider's environment settings.
- Restrict access to sensitive operational information.
- Validate incoming transaction requests.
- Review logs and monitor service health.

## Important Notes

- The project has been tested locally using Docker.
- Cloud deployment is a separate step and has not yet been completed.
- Cloud provider resource limits and pricing may change.
- The PaySim dataset is synthetic; production deployment requires further validation.
