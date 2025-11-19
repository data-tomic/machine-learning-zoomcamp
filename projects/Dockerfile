# 1. Use the official lightweight Python image
FROM python:3.12-slim

# 2. Set the working directory inside the container
WORKDIR /app

# 3. Copy the dependency file to the container
COPY requirements.txt .

# 4. Install dependencies
# --no-cache-dir is used to keep the image size small
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copy the model and the prediction script
COPY ["predict.py", "model.bin", "./"]

# 6. Expose the port we configured in the app (9697)
EXPOSE 9697

# 7. Define the command to run the application using Gunicorn
# predict:app -> file 'predict.py', instance 'app'
ENTRYPOINT ["gunicorn", "--bind", "0.0.0.0:9697", "predict:app"]
