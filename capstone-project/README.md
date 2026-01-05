# Leukemia Classification Service

This project classifies cell images as **Leukemia (ALL)** or **Normal (HEM)** using a Convolutional Neural Network (EfficientNet-B0).

## Problem Description
Acute Lymphoblastic Leukemia (ALL) is a cancer of the blood and bone marrow. Early diagnosis is crucial. This project uses computer vision to automate the detection of leukemia cells from microscopic images.
Dataset used: [C-NMC Leukemia Dataset](https://www.kaggle.com/datasets/andrewmvd/leukemia-classification).

## Project Structure
* `notebook.ipynb`: Exploratory Data Analysis (EDA) and model training experiments.
* `train.py`: Script to train the model and export it to ONNX format.
* `predict.py`: Flask web service for inference using ONNX Runtime.
* `Dockerfile`: Instructions for containerization.

## Setup

1. **Install dependencies:**
   ```bash
   pip install pipenv
   pipenv install
