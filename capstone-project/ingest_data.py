import os
import boto3
from kaggle.api.kaggle_api_extended import KaggleApi
from botocore.exceptions import ClientError
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import shutil

# --- КОНФИГУРАЦИЯ (Берем из ENV, как в твоем скрипте) ---
# Настройки MinIO
MINIO_ENDPOINT = os.getenv("MINIO_S3_ENDPOINT_URL", "https://s3.k8s.dgoi.ru")
ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY", "zOw8x0hri01phOFO5POr")
SECRET_KEY = os.getenv("MINIO_SECRET_KEY", "zi5HjnYiWZhn07IjrwpvL3wZoJ72JrR4YuyR63Nr")
BUCKET_NAME = os.getenv("MINIO_BUCKET_NAME", "leukemia-data")

# Настройки Kaggle Dataset
DATASET_NAME = "andrewmvd/leukemia-classification" # C-NMC dataset
LOCAL_CACHE_DIR = os.getenv("LOCAL_CACHE_DIR", "./temp_data")

def get_s3_client():
    return boto3.client('s3',
                        endpoint_url=MINIO_ENDPOINT,
                        aws_access_key_id=ACCESS_KEY,
                        aws_secret_access_key=SECRET_KEY,
                        verify=False)

def check_bucket_exists(s3, bucket_name):
    """Проверяет существование бакета и создает его, если нет."""
    try:
        s3.head_bucket(Bucket=bucket_name)
        print(f"✅ Bucket '{bucket_name}' exists.")
    except ClientError:
        print(f"⚠️ Bucket '{bucket_name}' not found. Creating...")
        try:
            s3.create_bucket(Bucket=bucket_name)
            print(f"✅ Bucket '{bucket_name}' created successfully.")
        except Exception as e:
            print(f"❌ Critical Error creating bucket: {e}")
            exit(1)

def is_dataset_in_minio(s3, bucket_name):
    """Проверяет, есть ли уже данные в MinIO (по наличию папки validation)."""
    # C-NMC структура: C-NMC_Leukemia/training_data/...
    # Проверим наличие хотя бы одного файла, чтобы не перезаливать зря
    result = s3.list_objects_v2(Bucket=bucket_name, Prefix="C-NMC_Leukemia/", MaxKeys=1)
    return 'Contents' in result

def download_dataset_from_kaggle():
    """Скачивает и распаковывает датасет локально."""
    print("⬇️ Downloading dataset from Kaggle...")
    api = KaggleApi()
    api.authenticate()

    if not os.path.exists(LOCAL_CACHE_DIR):
        os.makedirs(LOCAL_CACHE_DIR)

    # Проверка, скачано ли уже
    if os.path.exists(os.path.join(LOCAL_CACHE_DIR, "C-NMC_Leukemia")):
        print("✅ Dataset already downloaded locally.")
        return

    api.dataset_download_files(DATASET_NAME, path=LOCAL_CACHE_DIR, unzip=True)
    print("✅ Download and extraction complete.")

def upload_file_worker(args):
    """Функция для пула потоков."""
    s3_client, file_path, bucket, object_name = args
    try:
        s3_client.upload_file(file_path, bucket, object_name)
    except Exception as e:
        return f"Error uploading {object_name}: {e}"
    return None

def upload_to_minio(s3):
    """Загружает файлы из локальной папки в MinIO многопоточно."""
    print("🚀 Starting upload to MinIO...")
    
    files_to_upload = []
    # Рекурсивно проходим по скачанной папке
    for root, dirs, files in os.walk(LOCAL_CACHE_DIR):
        for file in files:
            local_path = os.path.join(root, file)
            # Убираем префикс temp_data из пути в бакете
            relative_path = os.path.relpath(local_path, LOCAL_CACHE_DIR)
            # Можно добавить префикс версии, если нужно (как в твоем скрипте PROJECT_ROOT)
            # object_name = f"raw_data/{relative_path}" 
            object_name = relative_path 
            files_to_upload.append((local_path, object_name))

    print(f"📦 Found {len(files_to_upload)} files to upload.")

    # Используем ThreadPoolExecutor для скорости
    # Передаем s3 client в каждый поток (boto3 client потокобезопасен)
    
    with ThreadPoolExecutor(max_workers=20) as executor:
        # Создаем список задач
        futures = []
        for local_path, object_name in files_to_upload:
            futures.append(executor.submit(upload_file_worker, (s3, local_path, BUCKET_NAME, object_name)))
        
        # Отображаем прогресс
        for future in tqdm(futures, total=len(files_to_upload), unit="file"):
            result = future.result()
            if result:
                print(result) # Печатаем ошибку, если была

    print("✅ Upload to MinIO complete!")

if __name__ == "__main__":
    print(f"🔌 Connecting to MinIO at {MINIO_ENDPOINT}...")
    s3 = get_s3_client()
    
    check_bucket_exists(s3, BUCKET_NAME)
    
    if is_dataset_in_minio(s3, BUCKET_NAME):
        print("✨ Dataset already exists in MinIO. Skipping download & upload.")
        # Можно добавить флаг --force для перезаливки
    else:
        download_dataset_from_kaggle()
        upload_to_minio(s3)
        
        # Очистка (опционально)
        # print("Cleaning up local cache...")
        # shutil.rmtree(LOCAL_CACHE_DIR)
        print("🎉 Done! Data is ready in MinIO.")
