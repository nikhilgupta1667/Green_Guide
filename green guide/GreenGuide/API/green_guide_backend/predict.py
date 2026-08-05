import torch
import torchvision.transforms as transforms
from PIL import Image

# Load your model (adjust architecture as per your training)
model_path = "model/plant_disease_model.pth"

# Dummy class names (update based on your dataset)
class_names = ["Healthy", "Powdery Mildew", "Rust", "Leaf Spot"]

# Load model
model = torch.load(model_path, map_location=torch.device("cpu"))
model.eval()

# Image transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def predict_image(image_path):
    try:
        image = Image.open(image_path).convert("RGB")
        input_tensor = transform(image).unsqueeze(0)  # Add batch dimension

        with torch.no_grad():
            output = model(input_tensor)
            probabilities = torch.nn.functional.softmax(output[0], dim=0)
            predicted_class = torch.argmax(probabilities).item()
            confidence = probabilities[predicted_class].item()
            return class_names[predicted_class], confidence

    except Exception as e:
        print(f"❌ Error in prediction: {e}")
        return "Unknown", 0.0
