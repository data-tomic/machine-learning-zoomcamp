# Используем официальный легкий образ Python
FROM python:3.12-slim

# Устанавливаем uv для быстрой установки зависимостей
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

# Устанавливаем рабочую директорию
WORKDIR /app

# Копируем файлы зависимостей
COPY pyproject.toml uv.lock ./

# Устанавливаем зависимости
# --frozen: использовать строго версии из lock-файла
# --no-dev: не ставить библиотеки для разработки
RUN uv sync --frozen --no-dev

# Копируем код сервиса и модель
COPY predict.py leukemia_model.onnx ./

# Открываем порт
EXPOSE 9696

# Запускаем приложение через Gunicorn
# uv run гарантирует, что используется виртуальное окружение
ENTRYPOINT ["uv", "run", "gunicorn", "--bind=0.0.0.0:9696", "predict:app"]
