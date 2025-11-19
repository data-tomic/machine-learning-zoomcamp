import requests

url = 'http://localhost:9697/predict'

# Example 1: High Hemoglobin (Should be Healthy / False)
patient_healthy = {
    'Gender': 0,
    'Hemoglobin': 15.2,
    'MCH': 24.0,
    'MCHC': 31.0,
    'MCV': 86.0
}

# Example 2: Low Hemoglobin (Should be Anemia / True)
patient_sick = {
    'Gender': 1,
    'Hemoglobin': 9.5,
    'MCH': 20.0,
    'MCHC': 28.0,
    'MCV': 75.0
}

print("Sending Healthy Patient data...")
response = requests.post(url, json=patient_healthy).json()
print(response)

print("\nSending Sick Patient data...")
response = requests.post(url, json=patient_sick).json()
print(response)
