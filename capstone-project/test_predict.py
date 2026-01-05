import pytest
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
