"""
Training script for Leukemia Classification Project.
This script performs the following steps:
1. Prepares the dataset (split into train/val).
2. Fine-tunes a pre-trained EfficientNet-B0 model.
3. Saves the PyTorch model (.pth).
4. Exports the model to ONNX format (.onnx) for production deployment.
"""

import os
import copy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from sklearn.model_selection import train_test_split
from PIL import Image
from tqdm import tqdm

# --- Configuration ---
# Path where ingest_data.py saved the images
DATA_DIR = "./temp_data/C-NMC_Leukemia" 
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model output filenames
MODEL_SAVE_PATH = "leukemia_model.pth"
ONNX_SAVE_PATH = "leukemia_model.onnx"

class LeukemiaDataset(Dataset):
    """Custom Dataset class for loading Leukemia images."""
    
    def __init__(self, file_paths, labels, transform=None):
        self.file_paths = file_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        path = self.file_paths[idx]
        label = self.labels[idx]
        try:
            # Open image and convert to RGB to ensure 3 channels
            image = Image.open(path).convert("RGB")
            if self.transform:
                image = self.transform(image)
            return image, label
        except Exception as e:
            print(f"Error loading {path}: {e}")
            # Return a black image in case of error to prevent crash
            return torch.zeros((3, IMG_SIZE, IMG_SIZE)), label

def prepare_data(data_dir):
    """
    Recursively scans the directory for images and splits them into train/val sets.
    """
    print("Scanning directory for data...")
    all_files = []
    labels = []

    # Recursively find files
    for root, _, files in os.walk(data_dir):
        for file in files:
            if file.lower().endswith(('.bmp', '.jpg', '.png')):
                full_path = os.path.join(root, file)
                
                # Assign labels based on folder names
                if '/all/' in full_path.lower():
                    all_files.append(full_path)
                    labels.append(1) # Label 1: Leukemia (ALL)
                elif '/hem/' in full_path.lower():
                    all_files.append(full_path)
                    labels.append(0) # Label 0: Normal (HEM)

    if len(all_files) == 0:
        raise ValueError(f"No images found in {data_dir}. Please run 'ingest_data.py' first.")

    print(f"Found {len(all_files)} images. Splitting data...")
    
    # Split: 80% Training, 20% Validation, Stratified by label
    return train_test_split(
        all_files, labels, test_size=0.2, random_state=42, stratify=labels
    )

def train_model(model, dataloaders, criterion, optimizer, num_epochs=5):
    """Main training loop."""
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data
            # Using tqdm for progress bar is recommended for long training
            for inputs, labels in tqdm(dataloaders[phase], desc=phase):
                inputs = inputs.to(DEVICE)
                labels = labels.to(DEVICE)

                optimizer.zero_grad()

                # Forward pass
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # Backward pass + Optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # Statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # Deep copy the model if it's the best so far
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
        
        print() # Empty line between epochs

    print(f'Training complete. Best Validation Accuracy: {best_acc:.4f}')
    
    # Load best model weights
    model.load_state_dict(best_model_wts)
    return model

def main():
    print(f"Using device: {DEVICE}")

    # 1. Data Preparation
    train_files, val_files, train_labels, val_labels = prepare_data(DATA_DIR)

    # 2. Define Transforms (ImageNet normalization)
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    # 3. Create Datasets and Dataloaders
    datasets = {
        'train': LeukemiaDataset(train_files, train_labels, data_transforms['train']),
        'val': LeukemiaDataset(val_files, val_labels, data_transforms['val'])
    }

    dataloaders = {
        x: DataLoader(datasets[x], batch_size=BATCH_SIZE, shuffle=(x=='train'), num_workers=2)
        for x in ['train', 'val']
    }

    # 4. Initialize Model (EfficientNet-B0)
    print("Initializing EfficientNet-B0 model...")
    model = models.efficientnet_b0(weights='DEFAULT')
    
    # Replace the classifier for 2 classes (Leukemia vs Normal)
    num_ftrs = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_ftrs, 2)
    model = model.to(DEVICE)

    # 5. Define Loss and Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # 6. Train
    model = train_model(model, dataloaders, criterion, optimizer, num_epochs=EPOCHS)

    # 7. Save PyTorch Model
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"PyTorch model saved to {MODEL_SAVE_PATH}")

    # 8. Export to ONNX (Crucial for lightweight deployment)
    print("Exporting model to ONNX...")
    model.eval()
    
    # Create a dummy input for tracing
    dummy_input = torch.randn(1, 3, IMG_SIZE, IMG_SIZE).to(DEVICE)
    
    torch.onnx.export(
        model,
        dummy_input,
        ONNX_SAVE_PATH,
        export_params=True,
        opset_version=12,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    )
    print(f"✅ ONNX model saved to {ONNX_SAVE_PATH}")

if __name__ == "__main__":
    main()
