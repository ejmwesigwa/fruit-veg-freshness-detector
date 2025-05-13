import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import numpy as np
import os
from grad_cam import GradCAM
from grad_cam.utils.model_targets import ClassifierOutputTarget
from grad_cam.utils.image import show_cam_on_image

# --- CONFIG ---
MODEL_PATH = 'models/best_resnet_model.pth'
CLASS_NAMES = [
    "Apple - Healthy", "Apple - Rotten",
    "Banana - Healthy", "Banana - Rotten",
    "Bellpepper - Healthy", "Bellpepper - Rotten",
    "Carrot - Healthy", "Carrot - Rotten",
    "Cucumber - Healthy", "Cucumber - Rotten",
    "Grape - Healthy", "Grape - Rotten",
    "Guava - Healthy", "Guava - Rotten",
    "Jujube - Healthy", "Jujube - Rotten",
    "Mango - Healthy", "Mango - Rotten",
    "Orange - Healthy", "Orange - Rotten",
    "Pomegranate - Healthy", "Pomegranate - Rotten",
    "Potato - Healthy", "Potato - Rotten",
    "Strawberry - Healthy", "Strawberry - Rotten",
    "Tomato - Healthy", "Tomato - Rotten"
] 

# --- Load model ---
@st.cache_resource
def load_model():
    model = models.resnet18(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, len(CLASS_NAMES))
    model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
    model.eval()
    return model

# --- Image Preprocessing ---
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])
    return transform(image).unsqueeze(0)

# --- Grad-CAM Visualization ---
def generate_gradcam(model, input_tensor, class_idx):
    target_layers = [model.layer4[-1]]
    cam = GradCAM(model=model, target_layers=target_layers)
    targets = [ClassifierOutputTarget(class_idx)]
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0, :]
    return grayscale_cam

# --- Main App ---
def main():
    st.set_page_config(page_title="Fruit & Veg Classifier", layout="centered")
    st.title("🥦 Fruit & Vegetable Freshness Classifier")
    st.write("Upload an image of a fruit or vegetable and get its type and freshness status.")

    model = load_model()

    uploaded_file = st.file_uploader("📤 Upload an image", type=['jpg', 'jpeg', 'png'])

    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_column_width=True)

        input_tensor = preprocess_image(image)
        outputs = model(input_tensor)
        _, pred = torch.max(outputs, 1)
        prediction = CLASS_NAMES[pred.item()]
        st.success(f"✅ Prediction: **{prediction}**")

        # Show Grad-CAM
        with st.expander("🔍 Show Grad-CAM Heatmap"):
            img_np = np.array(image.resize((224, 224))) / 255.0
            grayscale_cam = generate_gradcam(model, input_tensor, pred.item())
            cam_image = show_cam_on_image(img_np, grayscale_cam, use_rgb=True)
            st.image(cam_image, caption="Grad-CAM Visualization", use_column_width=True)

if __name__ == '__main__':
    main()
