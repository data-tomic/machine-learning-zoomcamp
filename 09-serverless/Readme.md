# ML Zoomcamp 2025 - Homework 9: Serverless Deep Learning

This repository contains the solution for Module 9 of the Machine Learning Zoomcamp.
Topic: Model Deployment (Serverless, AWS Lambda, Docker, ONNX).

## 🛠 Prerequisites & Setup

This project uses **uv** for dependency management and **Docker** for containerization.

1.  **Initialize environment:**
    ```bash
    uv sync
    ```
    *This installs `onnxruntime`, `pillow`, `numpy`, and `requests`.*

2.  **Model Files:**
    For local testing (Questions 1-4), the file `hair_classifier_v1.onnx` must be present in the root directory.

## 📂 File Description

*   `main.py` — The Lambda handler function. It loads the model, performs ImageNet preprocessing, and returns the prediction. Used inside the Docker container.
*   `Dockerfile` — Instructions to build the image based on `agrigorev/model-2025-hairstyle:v1`.
*   `q1_inspect.py` — Script to inspect the model's input and output names (Question 1).
*   `q2_predict.py` — Local script for preprocessing (resize + ImageNet normalization) and inference (Questions 2, 3, 4).
*   `q6_test.py` — Script to send a POST request to the running Docker container (Question 6).

---

## 🚀 Usage & Solutions

### Question 1: Inspect Model
Identify the output name of the model.
```bash
uv run python q1_inspect.py
