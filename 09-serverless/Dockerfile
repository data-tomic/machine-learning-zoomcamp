FROM agrigorev/model-2025-hairstyle:v1

# Устанавливаем библиотеки.
# В базовом образе уже есть модель, но нужен onnxruntime и pillow
# (возможно, они там уже есть, но лучше перестраховаться или использовать то, что нужно).
# Обычно в этом задании просят установить `https://github.com/alexeygrigorev/tflite-aws-lambda/raw/main/tflite/tflite_runtime-2.14.0-cp310-cp310-linux_x86_64.whl` если это tflite,
# но у нас ONNX. Значит ставим onnxruntime.

RUN pip install -U pip
RUN pip install onnxruntime pillow requests

COPY main.py .

# Указываем команду запуска (для AWS Lambda base images)
CMD [ "main.lambda_handler" ]
