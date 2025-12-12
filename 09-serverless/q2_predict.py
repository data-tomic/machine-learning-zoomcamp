import numpy as np
from PIL import Image
from io import BytesIO
import requests
import onnxruntime as ort

# --- Константы ---
MODEL_PATH = 'hair_classifier_v1.onnx'
IMAGE_URL = 'https://habrastorage.org/webt/yf/_d/ok/yf_dokzqy3vcritme8ggnzqlvwa.jpeg'
TARGET_SIZE = (200, 200)

def download_prepare_image(url):
    response = requests.get(url)
    img = Image.open(BytesIO(response.content))
    img = img.convert('RGB')
    img = img.resize(TARGET_SIZE, Image.NEAREST)
    return img

img = download_prepare_image(IMAGE_URL)

# --- Preprocessing (ImageNet) ---
x = np.array(img, dtype=np.float32) / 255.0

# Mean и Std для ImageNet
mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# Нормализация: (x - mean) / std
# Сначала вычтем mean и поделим на std (NumPy сделает это по последней оси каналов корректно)
x = (x - mean) / std

# Транспонируем в (3, 200, 200)
x = x.transpose(2, 0, 1)

# Проверяем первый пиксель (Red channel) для Вопроса 3
print(f"First pixel R value: {x[0, 0, 0]}")

# Добавляем батч
X = np.expand_dims(x, axis=0)

# --- Предсказание (Вопрос 4) ---
session = ort.InferenceSession(MODEL_PATH)
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

pred = session.run([output_name], {input_name: X})[0]

# Получаем raw logit
logit = pred[0][0]
print(f"Raw logit: {logit}")

# Считаем вероятность (Sigmoid)
probability = 1 / (1 + np.exp(-logit))
print(f"Output probability: {probability}")
