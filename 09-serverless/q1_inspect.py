import onnxruntime as ort

model_path = 'hair_classifier_v1.onnx'
session = ort.InferenceSession(model_path)

print("--- Outputs ---")
for o in session.get_outputs():
    print(f"Name: {o.name}, Shape: {o.shape}")
