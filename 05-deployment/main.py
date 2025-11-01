import pickle
from fastapi import FastAPI
from pydantic import BaseModel

# Загружаем модель
# with open('pipeline_v1.bin', 'rb') as f_in:
    # pipeline = pickle.load(f_in)

with open('/code/pipeline_v2.bin', 'rb') as f_in:
    pipeline = pickle.load(f_in)

# Создаем FastAPI приложение
app = FastAPI()

# Определяем, какие данные мы ожидаем на входе
class ClientData(BaseModel):
    lead_source: str
    number_of_courses_viewed: int
    annual_income: float

# Создаем endpoint для предсказаний, который принимает POST-запросы
@app.post("/predict")
def predict(client: ClientData):
    # Преобразуем полученные данные в словарь
    client_dict = client.model_dump()
    
    # Делаем предсказание
    # Модель ожидает на вход список, поэтому передаем [client_dict]
    prediction = pipeline.predict_proba([client_dict])[0, 1]
    
    # Возвращаем результат в виде JSON
    return {"conversion_probability": prediction}
