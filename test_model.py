
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import pandas as pd

# Load breed names
df = pd.read_csv("data/dataset/labels.csv")
breeds = sorted(df["breed"].unique())

# Create ResNet50 model
model = models.resnet50(weights=None)
model.fc = nn.Linear(model.fc.in_features, 120)

# Load fine-tuned weights
state = torch.load(
    "dog_breed_model_ft.pth",
    map_location="cpu",
    weights_only=True
)

model.load_state_dict(state)
model.eval()

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# Select a training image for a smoke test
image_name = df.iloc[0]["id"]
image_path = f"data/dataset/train/{image_name}.jpg"

image = Image.open(image_path).convert("RGB")
image = transform(image).unsqueeze(0)

# Predict
with torch.no_grad():
    output = model(image)
    probabilities = torch.softmax(output, dim=1)
    confidence, predicted = torch.max(probabilities, 1)

print("Actual breed:", df.iloc[0]["breed"])
print("Predicted breed:", breeds[predicted.item()])
print("Confidence:", round(confidence.item() * 100, 2), "%")
