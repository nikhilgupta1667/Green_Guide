# utils/predict.py
import torch
from torchvision import models, transforms
from PIL import Image

# Load MobileNetV2
model = models.mobilenet_v2(pretrained=True)
model.eval()

# Preprocess image
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Load ImageNet class labels
with open("utils/imagenet_classes.txt") as f:
    class_names = [line.strip() for line in f.readlines()]

def predict_image_class(image_path: str) -> str:
    image = Image.open(image_path).convert("RGB")
    input_tensor = preprocess(image).unsqueeze(0)  # Add batch dimension
    with torch.no_grad():
        output = model(input_tensor)
        _, predicted = output.max(1)
        return class_names[predicted.item()]
