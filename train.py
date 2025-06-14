# -------------------------
# 1. Imports and Setup
# -------------------------
import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
from sklearn.model_selection import train_test_split
from torchvision.models.vision_transformer import vit_b_16

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------
# 2. Dataset Path
# -------------------------
DATA_DIR = "/kaggle/input/skin-dataset/train"
benign_dir = os.path.join(DATA_DIR, "benign")
malignant_dir = os.path.join(DATA_DIR, "malignant")

benign_paths = [os.path.join(benign_dir, f) for f in os.listdir(benign_dir)
                if f.lower().endswith((".jpg", ".png", ".jpeg"))]
malignant_paths = [os.path.join(malignant_dir, f) for f in os.listdir(malignant_dir)
                   if f.lower().endswith((".jpg", ".png", ".jpeg"))]

# Oversample malignant to match benign
malignant_paths = malignant_paths * (len(benign_paths) // len(malignant_paths))

image_paths = benign_paths + malignant_paths
labels = [0] * len(benign_paths) + [1] * len(malignant_paths)

# Shuffle data
combined = list(zip(image_paths, labels))
random.shuffle(combined)
image_paths[:], labels[:] = zip(*combined)

# Split train/val
train_paths, val_paths, train_labels, val_labels = train_test_split(
    image_paths, labels, test_size=0.2, random_state=42, stratify=labels)

# -------------------------
# 3. Transforms
# -------------------------
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# -------------------------
# 4. Custom Dataset
# -------------------------
class SkinLesionDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label

# Datasets and Loaders
train_dataset = SkinLesionDataset(train_paths, train_labels, train_transform)
val_dataset = SkinLesionDataset(val_paths, val_labels, val_transform)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

# -------------------------
# 5. Vision Transformer Model
# -------------------------
model = vit_b_16(pretrained=True)
model.heads.head = nn.Linear(model.heads.head.in_features, 2)
model.to(device)

# -------------------------
# 6. Training Setup
# -------------------------
# Weighted Loss
weights = torch.tensor([1.0, 3.0]).to(device)
criterion = nn.CrossEntropyLoss(weight=weights)

# Optimizer and Scheduler
optimizer = optim.Adam(model.parameters(), lr=1e-4)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

# -------------------------
# 7. Training Function
# -------------------------
def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, device, epochs=30):
    best_val_acc = 0.0

    for epoch in range(epochs):
        model.train()
        train_loss, train_correct = 0.0, 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)
            preds = torch.argmax(outputs, dim=1)
            train_correct += (preds == labels).sum().item()

        train_loss /= len(train_loader.dataset)
        train_acc = train_correct / len(train_loader.dataset)

        # Validation phase
        model.eval()
        val_loss, val_correct = 0.0, 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
                preds = torch.argmax(outputs, dim=1)
                val_correct += (preds == labels).sum().item()

        val_loss /= len(val_loader.dataset)
        val_acc = val_correct / len(val_loader.dataset)

        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f} - Train Acc: {train_acc:.4f} - Val Loss: {val_loss:.4f} - Val Acc: {val_acc:.4f}")

        scheduler.step()

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "best_model.pth")
            print("✅ Best model saved!")

# -------------------------
# 8. Run Training
# -------------------------
train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, device, epochs=30)
print("\nTraining completed!")
