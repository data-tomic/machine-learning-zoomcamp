import pytest
import io
import numpy as np
from PIL import Image
from predict import app

@pytest.fixture
def client():
    # Создаем тестовый клиент Flask
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

def test_health_check(client):
    """Проверяем, что сервис жив"""
    response = client.get('/health')
    assert response.status_code == 200
    assert response.json == {'status': 'ok'}

def test_model_loading():
    """Проверяем, что глобальная сессия ONNX инициализирована"""
    from predict import ORT_SESSION
    assert ORT_SESSION is not None

def test_prediction_endpoint(client):
    """
    Интеграционный тест: отправляем сгенерированную картинку
    и проверяем, что модель возвращает корректный JSON.
    """
    # 1. Генерируем случайную картинку 224x224 (шум)
    # 3 канала (RGB), uint8 (0-255)
    random_array = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    img = Image.fromarray(random_array)
    
    # Сохраняем её в байты (как будто это файл)
    img_bytes = io.BytesIO()
    img.save(img_bytes, format='BMP')
    img_bytes.seek(0) # Перематываем в начало

    # 2. Отправляем POST запрос
    data = {'file': (img_bytes, 'test_cell.bmp')}
    response = client.post('/predict', data=data, content_type='multipart/form-data')

    # 3. Проверки
    assert response.status_code == 200
    json_data = response.get_json()
    
    # Проверяем структуру ответа
    assert 'prediction' in json_data
    assert 'probability' in json_data
    assert 'details' in json_data
    
    # Проверяем, что вероятность валидная (0..1)
    assert 0 <= json_data['probability'] <= 1
    # Проверяем, что предсказание - это один из наших классов
    assert json_data['prediction'] in ['Leukemia', 'Normal']
