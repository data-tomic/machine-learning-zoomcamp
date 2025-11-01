# Используем базовый образ
FROM agrigorev/zoomcamp-model:2025

# Устанавливаем рабочую директорию
WORKDIR /app

# Устанавливаем uv
RUN pip install uv

# Копируем файлы проекта
COPY pyproject.toml uv.lock ./

# Устанавливаем зависимости в системный Python контейнера.
# Флаг --system обязателен для uv в такой среде.
RUN uv pip install --system .

# Копируем наш FastAPI скрипт
COPY main.py .

# Открываем порт
EXPOSE 9696

# Команда для запуска веб-сервиса
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "9696"]
