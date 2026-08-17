"""
Image classifier with transfer learning (PyTorch)
============================================================
Usage:
    1. Organize your images into dataset/train/<class>/*.jpg and dataset/val/<class>/*.jpg
    2. Run: python train_classifier.py
    3. The trained model is saved as "model.pth"

With only a few hundred images per class, transfer learning
(starting from a model pretrained on ImageNet) works much better
than training a network from scratch.
"""

import torch
import torch.nn as nn
from torch.optim import Adam
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

# ---------------------------------------------------------
# 1. Configuration
# ---------------------------------------------------------
DATA_DIR = "dataset"          # folder containing train/ and val/ subfolders
BATCH_SIZE = 16
EPOCHS = 15
LEARNING_RATE = 0.001
IMG_SIZE = 224                 # size expected by ResNet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ---------------------------------------------------------
# 2. Image transforms
# ---------------------------------------------------------
# Training uses data augmentation (random variations) to help
# the model generalize better with a limited amount of data.
train_transforms = transforms.Compose([
    transforms.RandomResizedCrop(IMG_SIZE),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                          std=[0.229, 0.224, 0.225]),  # ImageNet statistics
])

val_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                          std=[0.229, 0.224, 0.225]),
])

# ---------------------------------------------------------
# 3. Load dataset
# ---------------------------------------------------------
train_dataset = datasets.ImageFolder(f"{DATA_DIR}/train", transform=train_transforms)
val_dataset = datasets.ImageFolder(f"{DATA_DIR}/val", transform=val_transforms)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

class_names = train_dataset.classes
num_classes = len(class_names)
print(f"Classes found ({num_classes}): {class_names}")

# ---------------------------------------------------------
# 4. Model: pretrained ResNet18, with the final layer
#    replaced to match our number of classes
# ---------------------------------------------------------
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

# Freeze all layers by default (they already know how to detect
# general shapes/edges from ImageNet pretraining)
for param in model.parameters():
    param.requires_grad = False

# Unfreeze the last convolutional block (layer4) so it can adapt
# to features specific to your classes, not just generic ones
for param in model.layer4.parameters():
    param.requires_grad = True

# Replace the final classification layer, which is always trained.
# A dropout layer is added before it to reduce overfitting risk,
# since we're now training more parameters (layer4 + fc) on a
# relatively small dataset.
model.fc = nn.Sequential(
    nn.Dropout(0.3),
    nn.Linear(model.fc.in_features, num_classes),
)
model = model.to(device)

criterion = nn.CrossEntropyLoss()

# Use a lower learning rate for the unfrozen pretrained layer4
# (it's already fairly well-tuned, so we adjust it gently) and a
# higher one for the brand-new final layer (which starts from
# random weights and needs to learn faster)
optimizer = Adam([
    {"params": model.layer4.parameters(), "lr": LEARNING_RATE / 10},
    {"params": model.fc.parameters(), "lr": LEARNING_RATE},
])

# Reduces the learning rate when validation accuracy stops improving,
# which helps stabilize training once the model starts oscillating
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="max", factor=0.5, patience=2
)

# ---------------------------------------------------------
# 5. Training loop
# ---------------------------------------------------------
def evaluate():
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct / total if total > 0 else 0.0


best_val_acc = 0.0
best_state = None

for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    val_acc = evaluate()
    scheduler.step(val_acc)
    print(f"Epoch {epoch+1}/{EPOCHS} - loss: {running_loss/len(train_loader):.4f} "
          f"- validation accuracy: {val_acc*100:.1f}%")

    # Keep track of the best model seen so far (avoids overfitting
    # by not blindly saving whatever the last epoch produced)
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_state = {k: v.clone() for k, v in model.state_dict().items()}
        print(f"  -> New best model (accuracy: {best_val_acc*100:.1f}%)")

# ---------------------------------------------------------
# 6. Save the BEST model (not necessarily the last epoch)
# ---------------------------------------------------------
torch.save({
    "model_state": best_state,
    "class_names": class_names,
}, "model.pth")

print(f"\nBest model saved as 'model.pth' (validation accuracy: {best_val_acc*100:.1f}%)")
