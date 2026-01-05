import numpy as np
import onnxruntime as ort
from flask import Flask, request, jsonify
from PIL import Image
import io

app = Flask(__name__)

# Load ONNX Model
ONNX_MODEL_PATH = "leukemia_model.onnx"
session = ort.InferenceSession(ONNX_MODEL_PATH)
input_name = session.get_inputs()[0].name

def preprocess_image(image_bytes):
    """
    Same preprocessing as in training: Resize(224) -> Normalize(ImageNet)
    """
    img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    img = img.resize((224, 224))
    
    x = np.array(img, dtype=np.float32) / 255.0
    
    # ImageNet Mean and Std
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    x = (x - mean) / std
    
    # Transpose (H, W, C) -> (C, H, W)
    x = x.transpose(2, 0, 1)
    
    # Add batch dimension -> (1, C, H, W)
    x = np.expand_dims(x, axis=0)
    return x

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    try:
        # Preprocess
        input_tensor = preprocess_image(file.read())
        
        # Inference ONNX
        outputs = session.run(None, {input_name: input_tensor})
        logits = outputs[0][0]
        
        # Softmax
        probs = np.exp(logits) / np.sum(np.exp(logits))
        
        # Classes: 0 = Normal, 1 = Leukemia (based on train.py sorting)
        classes = ['Normal (HEM)', 'Leukemia (ALL)']
        pred_idx = np.argmax(probs)
        
        result = {
            'class': classes[pred_idx],
            'probability_leukemia': float(probs[1]),
            'probability_normal': float(probs[0])
        }
        return jsonify(result)

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9696)
