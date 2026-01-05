import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from sklearn.model_selection import train_test_split
from PIL import Image
import copy

# --- Config ---
# Assuming data is already downloaded via ingest_data.py to ./data
DATA_DIR = "./temp_data/C-NMC_Leukemia" 
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_SAVE_PATH = "leukemia_model.pth"
ONNX_SAVE_PATH = "leukemia_model.onnx"

class LeukemiaDataset(Dataset):
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
            image = Image.open(path).convert("RGB")
            if self.transform:
                image = self.transform(image)
            return image, label
        except Exception as e:
            print(f"Error loading {path}: {e}")
            return torch.zeros((3, IMG_SIZE, IMG_SIZE)), label

def prepare_data(data_dir):
    print("Preparing data file lists...")
    all_files = []
    labels = []
    
    # Recursively find files
    for root, _, files in os.walk(data_dir):
        for file in files:
            if file.lower().endswith(('.bmp', '.jpg', '.png')):
                full_path = os.path.join(root, file)
                if '/all/' in full_path.lower():
                    all_files.append(full_path)
                    labels.append(1) # Leukemia
                elif '/hem/' in full_path.lower():
                    all_files.append(full_path)
                    labels.append(0) # Normal
    
    if len(all_files) == 0:
        raise ValueError(f"No images found in {data_dir}. Please run ingest_data.py first.")

    return train_test_split(all_files, labels, test_size=0.2, random_state=42, stratify=labels)

def train_model(model, dataloaders, criterion, optimizer, num_epochs=5):
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(DEVICE)
                labels = labels.to(DEVICE)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

    print(f'Best Val Acc: {best_acc:.4f}')
    model.load_state_dict(best_model_wts)
    return model

def main():
    print(f"Using device: {DEVICE}")
    
    train_files, val_files, train_labels, val_labels = prepare_data(DATA_DIR)
    
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

    datasets = {
        'train': LeukemiaDataset(train_files, train_labels, data_transforms['train']),
        'val': LeukemiaDataset(val_files, val_labels, data_transforms['val'])
    }
    
    dataloaders = {x: DataLoader(datasets[x], batch_size=BATCH_SIZE, shuffle=(x=='train'), num_workers=2)
                   for x in ['train', 'val']}

    # Initialize EfficientNet
    model = models.efficientnet_b0(weights='DEFAULT')
    num_ftrs = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_ftrs, 2)
    model = model.to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # Train
    model = train_model(model, dataloaders, criterion, optimizer, num_epochs=EPOCHS)

    # Save PyTorch Model
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"Model saved to {MODEL_SAVE_PATH}")

    # Export to ONNX (Crucial for deployment)
    model.eval()
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
    print(f"ONNX model saved to {ONNX_SAVE_PATH}")

if __name__ == "__main__":
    main()
