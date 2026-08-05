from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
import shutil
from PIL import Image
import torch
import torchvision.transforms as transforms
from torchvision import models
from fpdf import FPDF
import uuid

app = FastAPI()

# CORS Middleware for frontend connection
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # or restrict to your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = Path("static/uploads")
REPORT_DIR = Path("static/reports")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

# Load MobileNet model
model = models.mobilenet_v2(pretrained=True)
model.eval()

# Transform to preprocess image
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Load ImageNet labels
with open("imagenet_classes.txt") as f:
    imagenet_labels = [line.strip() for line in f.readlines()]

@app.post("/upload")
async def upload_image(
    plantImage: UploadFile = File(...),
    location: str = Form(...),
    plantType: str = Form(None)
):
    try:
        # Save uploaded image
        image_id = str(uuid.uuid4())
        filename = f"{image_id}_{plantImage.filename}"
        file_path = UPLOAD_DIR / filename

        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(plantImage.file, buffer)

        # Open image and prepare tensor
        image = Image.open(file_path).convert("RGB")
        image_tensor = transform(image).unsqueeze(0)

        # Predict using MobileNet
        with torch.no_grad():
            outputs = model(image_tensor)
            _, predicted = outputs.max(1)
            predicted_label = imagenet_labels[predicted.item()]

        # Generate PDF report with uploaded image
        report_path = REPORT_DIR / f"{image_id}_report.pdf"
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=14)
        pdf.cell(200, 10, txt="Green Guide - Plant Health Report", ln=True, align='C')
        pdf.ln(10)
        pdf.cell(200, 10, txt=f"Prediction: {predicted_label}", ln=True)
        pdf.cell(200, 10, txt=f"Location: {location}", ln=True)
        pdf.cell(200, 10, txt=f"Plant Type: {plantType}", ln=True)
        pdf.ln(10)

        # Create a resized temp copy for PDF
        try:
            resized_path = str(file_path.with_suffix(".pdf_image.jpg"))
            resized_img = image.copy()
            resized_img.thumbnail((150, 150))  # Resize to fit in PDF
            resized_img.save(resized_path)

            # Add image to PDF
            pdf.image(resized_path, x=30, y=pdf.get_y(), w=100)
        except Exception as img_err:
            print(f"Image not added to PDF: {img_err}")

        pdf.output(str(report_path))

        return {
            "message": "Prediction successful",
            "report_url": f"http://127.0.0.1:8000/static/reports/{report_path.name}",
            "prediction": predicted_label
        }

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
