```markdown
# Credit Scoring MLOps System

This project is a microservice-based system designed for evaluating client creditworthiness (credit scoring). The system includes an automated backend with an ML model and an interactive web interface for end-users.

## Tech Stack

* **Model:** LightGBM (LGBMClassifier).
* **ML Lifecycle Management:** MLflow (Tracking, Artifact Store).
* **Backend:** FastAPI (Uvicorn).
* **Frontend:** Streamlit.
* **Infrastructure:** Docker, Docker Compose.
* **Runtime:** Python 3.12-slim.

## System Architecture

The system is divided into two isolated containers:
1. **Backend (Port 666):** Handles HTTP requests, loads model weights directly from the local MLflow storage (`mlruns`), and returns predictions in JSON format.
2. **Frontend (Port 777):** A web interface that allows users to input financial and personal data. It communicates with the backend via the internal Docker network.

## Requirements

To run the system, you must have the following installed:
* Docker
* Docker Compose

## Getting Started

1. Ensure that the `mlruns` folder containing the trained model is located in the project's root directory.
2. Open a terminal in the project directory and run the command to build and start the containers:

```bash
docker-compose up --build
```

3. Once launched, the services will be available at:
   * **Web Interface (Streamlit):** `http://localhost:777`
   * **API (FastAPI):** `http://localhost:666`

## Resource Limits

The `docker-compose.yml` configuration includes limits to prevent excessive consumption of host system resources:
* **Backend:** Up to 1 GB RAM, 2 CPU cores.
* **Frontend:** Up to 512 MB RAM, 2 CPU cores.

## Test Input Data

The model is trained on the German Credit dataset. Below are examples of valid input values:

| Parameter | Description | Example Value |
| :--- | :--- | :--- |
| Status of checking account | Account state | A11 (balance < 0 DM) |
| Duration in month | Loan term | 24 |
| Credit history | Credit history | A32 (existing credits paid back) |
| Credit amount | Loan amount | 3000 |
| Savings account/bonds | Savings | A61 ( < 100 DM) |
| Present employment since | Employment duration | A73 (1 <= ... < 4 years) |
| Age in years | Age | 35 |

## Project Structure

* `main.py` — FastAPI server logic and model initialization.
* `frontend.py` — Streamlit interface and request handling logic.
* `train.py` — Model training script with MLflow logging.
* `Dockerfile` / `Dockerfile.frontend` — Docker build instructions.
* `docker-compose.yml` — Orchestration and network configuration.
* `mlruns/` — MLflow artifact and metadata storage.
```