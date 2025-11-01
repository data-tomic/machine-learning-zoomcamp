import pickle

# Загружаем модель
with open('pipeline_v1.bin', 'rb') as f_in:
    pipeline = pickle.load(f_in)

# Данные для предсказания
record = {
    "lead_source": "paid_ads",
    "number_of_courses_viewed": 2,
    "annual_income": 79276.0
}

# Получаем вероятность
prediction = pipeline.predict_proba([record])[0, 1]

print(f"Вероятность конверсии: {prediction:.3f}")
