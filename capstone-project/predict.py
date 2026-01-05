import io
import numpy as np
import onnxruntime as ort
from flask import Flask, request, jsonify
from PIL import Image

app = Flask(__name__)

# --- Load Model ---
MODEL_PATH = "leukemia_model.onnx"
# Переименовали в ORT_SESSION (как ждет тест)
ORT_SESSION = ort.InferenceSession(MODEL_PATH)
INPUT_NAME = ORT_SESSION.get_inputs()[0].name

def preprocess_image(image_bytes):
    """
    Подготовка изображения: Resize -> Normalize -> Transpose
    """
    img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    img = img.resize((224, 224))
    
    # Normalize (ImageNet stats)
    img_data = np.array(img).astype('float32') / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img_data = (img_data - mean) / std
    
    # Transpose (H, W, C) -> (C, H, W)
    img_data = img_data.transpose(2, 0, 1)
    
    # Add Batch dimension
    img_data = np.expand_dims(img_data, axis=0)
    return img_data

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    try:
        input_tensor = preprocess_image(file.read())
        
        # Inference using global ORT_SESSION
        outputs = ORT_SESSION.run(None, {INPUT_NAME: input_tensor})
        logits = outputs[0][0]
        
        # Softmax
        probs = np.exp(logits) / np.sum(np.exp(logits))
        
        # Classes
        classes = ['Normal', 'Leukemia']
        pred_idx = np.argmax(probs)
        
        return jsonify({
            'prediction': classes[pred_idx],
            'probability': float(probs[pred_idx]),
            'details': {
                'normal_prob': float(probs[0]),
                'leukemia_prob': float(probs[1])
            }
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# --- Health Check (Добавили для тестов) ---
@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok'})

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9696)
