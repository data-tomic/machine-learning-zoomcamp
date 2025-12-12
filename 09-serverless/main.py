import numpy as np
import onnxruntime as ort
from PIL import Image
from io import BytesIO
import os
import requests

# ! ИЗМЕНЕНИЕ: Имя модели внутри контейнера другое
MODEL_PATH = os.getenv('MODEL_PATH', 'hair_classifier_empty.onnx')

session = ort.InferenceSession(MODEL_PATH)
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

def prepare_image(img, target_size=(200, 200)):
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img = img.resize(target_size, Image.NEAREST)
    return img

def preprocess(img):
    x = np.array(img, dtype=np.float32) / 255.0
    
    # Используем ту же нормализацию (ImageNet), что и раньше
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    x = (x - mean) / std
    
    x = x.transpose(2, 0, 1)
    x = np.expand_dims(x, axis=0)
    return x

def lambda_handler(event, context):
    url = event['url']
    img_data = requests.get(url).content
    img = Image.open(BytesIO(img_data))
    
    img = prepare_image(img)
    X = preprocess(img)
    
    pred = session.run([output_name], {input_name: X})[0]
    result = float(pred[0][0])
    
    return result
