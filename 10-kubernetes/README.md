# Machine Learning Zoomcamp 2025 - Homework 10: Kubernetes

This folder contains the solution for Module 10 of the [DataTalksClub Machine Learning Zoomcamp](https://github.com/DataTalksClub/machine-learning-zoomcamp). The goal is to containerize a churn prediction model and deploy it to a local Kubernetes cluster using **Kind**.

## Project Structure

*   `q6_predict.py`: The Flask application that serves the model.
*   `q6_test.py`: A test script to send a POST request to the running service.
*   `Dockerfile_full`: The Dockerfile used to build the image.
*   `pipeline_v2.bin`: The pre-trained model and pipeline file.
*   `pyproject.toml` / `uv.lock`: Python dependency definitions (managed by `uv`).

## Prerequisites

*   **Docker**
*   **Python 3.10+**
*   **Kind** (Kubernetes in Docker)
*   **Kubectl**
*   **uv** (Optional, for local dependency management)

## 1. Local Setup

If you want to run the code locally before containerizing:

```bash
# Install dependencies using uv
uv sync

# Or install manually via pip (if not using uv)
pip install flask gunicorn requests scikit-learn
