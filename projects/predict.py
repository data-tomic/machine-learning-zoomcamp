import pickle
from flask import Flask, request, jsonify

MODEL_FILE = 'model.bin'

# Load model and vectorizer
print(f"Loading model from {MODEL_FILE}...")
with open(MODEL_FILE, 'rb') as f_in:
    dv, model = pickle.load(f_in)

app = Flask('anemia-prediction')

@app.route('/predict', methods=['POST'])
def predict():
    patient_data = request.get_json()

    # Transform data
    X = dv.transform([patient_data])
    
    # Predict
    y_pred = model.predict_proba(X)[0, 1]
    anemia_detected = y_pred >= 0.5

    result = {
        'anemia_probability': float(y_pred),
        'anemia_detected': bool(anemia_detected)
    }

    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9697)
