import os
import shutil
from kaggle.api.kaggle_api_extended import KaggleApi

# --- Configuration ---
DATASET_NAME = "andrewmvd/leukemia-classification" # Официальное название датасета
LOCAL_DATA_DIR = "./temp_data"

def download_data_from_kaggle():
    """Downloads dataset from Kaggle using API."""
    print(f"⬇️  Downloading dataset '{DATASET_NAME}' from Kaggle...")
    
    # Проверка переменных окружения
    if not os.getenv("KAGGLE_USERNAME") or not os.getenv("KAGGLE_KEY"):
        print("❌ Error: Kaggle credentials not found in environment variables!")
        print("Please run:")
        print("  export KAGGLE_USERNAME=your_username")
        print("  export KAGGLE_KEY=your_key")
        return

    try:
        api = KaggleApi()
        api.authenticate()
        
        # Скачиваем и распаковываем
        if not os.path.exists(LOCAL_DATA_DIR):
            os.makedirs(LOCAL_DATA_DIR)
            
        print("📦 Downloading and unzipping... (This may take a minute)")
        api.dataset_download_files(DATASET_NAME, path=LOCAL_DATA_DIR, unzip=True)
        
        print(f"✅ Download complete! Data saved to {LOCAL_DATA_DIR}")
        
        # Проверка структуры (опционально, для отладки)
        if os.path.exists(os.path.join(LOCAL_DATA_DIR, "C-NMC_Leukemia")):
            print("📂 Structure verified: 'C-NMC_Leukemia' folder found.")
        else:
            print("⚠️ Warning: Unexpected folder structure.")

    except Exception as e:
        print(f"❌ Failed to download data: {e}")

if __name__ == "__main__":
    download_data_from_kaggle()
