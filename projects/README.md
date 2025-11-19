# Anemia Diagnosis Prediction 🩸

## Problem Description
Anemia is a condition in which you lack enough healthy red blood cells to carry adequate oxygen to your body's tissues. This project aims to predict whether a patient is likely to have anemia based on their blood test results (CBC features like Hemoglobin, MCH, MCHC, MCV).

The dataset was obtained from Kaggle/Open Sources. The project is part of the **Machine Learning Zoomcamp** Midterm Project.

## Dataset
The dataset contains the following features:
*   **Gender**: 0 or 1
*   **Hemoglobin**: Hemoglobin level
*   **MCH**: Mean Corpuscular Hemoglobin
*   **MCHC**: Mean Corpuscular Hemoglobin Concentration
*   **MCV**: Mean Corpuscular Volume
*   **Result**: Target variable (0 = Healthy, 1 = Anemia)

## Project Structure
*   `notebook.ipynb`: Data preparation, EDA, feature importance analysis, and model selection.
*   `train.py`: Script to train the final Logistic Regression model and save it to `model.bin`.
*   `predict.py`: Flask application to serve the model via HTTP API.
*   `Dockerfile`: Configuration to containerize the application.

## How to Run

### 1. Clone the repository
```bash
git clone git@github.com:data-tomic/machine-learning-zoomcamp.git
cd projects
```

### 2. Run with Docker (Recommended)
Build the image:
```bash
docker build -t anemia-app .
```

Run the container:
```bash
docker run -it --rm -p 9697:9697 anemia-app
```

### 3. Test the service
Open a separate terminal and run:
```bash
python test.py
```
You should see a JSON response indicating whether anemia was detected.

### 4. Run locally (without Docker)
If you prefer `venv`:
```bash
pip install -r requirements.txt
python train.py
python predict.py
```
