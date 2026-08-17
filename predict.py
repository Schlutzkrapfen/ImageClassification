"""
Use the trained model to classify a new image.

Usage:
    python predict.py path/to/image.jpg
"""

import sys
import torch
from torch import nn
from torchvision import transforms, models
from PIL import Image

IMG_SIZE = 224
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the saved model
checkpoint = torch.load("model.pth", map_location=device)
class_names = checkpoint["class_names"]

model = models.resnet18(weights=None)
model.fc = nn.Sequential(
    nn.Dropout(0.3),
    nn.Linear(model.fc.in_features, len(class_names)),
)
model.load_state_dict(checkpoint["model_state"])
model = model.to(device)
model.eval()

transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                          std=[0.229, 0.224, 0.225]),
])

image_path = sys.argv[1]
image = Image.open(image_path).convert("RGB")
input_tensor = transform(image).unsqueeze(0).to(device)

with torch.no_grad():
    outputs = model(input_tensor)
    probs = torch.softmax(outputs, dim=1)[0]
    predicted_idx = torch.argmax(probs).item()

print(f"Predicted class: {class_names[predicted_idx]}")
print(f"Confidence: {probs[predicted_idx]*100:.1f}%"):w

