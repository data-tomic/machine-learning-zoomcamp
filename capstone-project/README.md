# Leukemia Classification Service

This project classifies microscopic images of blood cells into two categories: **Normal (HEM)** and **Leukemia (ALL)** using a Convolutional Neural Network (EfficientNet-B0).

The project is implemented using **PyTorch** for training and **ONNX Runtime** for high-performance inference. Dependency management is handled by **uv**.

## Problem Description
Acute Lymphoblastic Leukemia (ALL) is a cancer of the blood and bone marrow. Early diagnosis is crucial. This project uses computer vision to automate the detection of leukemia cells.
Dataset: [C-NMC Leukemia Dataset](https://www.kaggle.com/datasets/andrewmvd/leukemia-classification).

## Technologies
*   **Training**: PyTorch, EfficientNet-B0 (Transfer Learning)
*   **Inference**: ONNX Runtime (CPU optimized)
*   **Containerization**: Docker
*   **Package Manager**: uv
*   **Web Framework**: Flask + Gunicorn

## Project Structure
*   `notebook.ipynb`: EDA, visualization, and experiments.
*   `ingest_data.py`: Script to download dataset from Kaggle/S3.
*   `train.py`: Training pipeline (trains on GPU -> exports .onnx).
*   `predict.py`: Inference service.
*   `Dockerfile`: Multi-stage build instructions.

## How to Run

### 1. Prerequisites
Install [uv](https://github.com/astral-sh/uv):
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 2. Setup & Training
Clone the repo and sync dependencies:
```bash
uv sync
```

Download the data (Requires Kaggle API keys in env vars or `~/.kaggle/kaggle.json`):
```bash
uv run ingest_data.py
```

Train the model (GPU recommended):
```bash
uv run train.py
```
*This will generate `leukemia_model.onnx`.*

### 3. Docker Deployment
Build the image:
```bash
docker build -t leukemia-service .
```

Run the container:
```bash
docker run -it --rm -p 9696:9696 leukemia-service
```

### 4. Usage
Send a POST request with a cell image (`.bmp`, `.jpg`, `.png`):

```bash
curl -X POST -F "file=@path/to/image.bmp" http://localhost:9696/predict
```

**Response Example:**
```json
{
  "prediction": "Leukemia (ALL)",
  "probability": 0.995,
  "details": {
    "leukemia_prob": 0.995,
    "normal_prob": 0.005
  }
}
```
