import streamlit as st
from PIL import Image
import torch
import torch.nn as nn
from torchvision import models, transforms
import os
import requests

# Configuration
MODEL_URL = "https://github.com/Taneesha3105/PCOS_detection/releases/download/v1.0.0/PCOS_resnet18_model.pth"
MODEL_PATH = "PCOS_resnet18_model.pth"
CLASS_NAMES = ['No PCOS', 'PCOS']
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Streamlit page setup
st.set_page_config(page_title="PCOS Companion", page_icon="💖", layout="centered")

# Download model if not present
if not os.path.exists(MODEL_PATH):
    with st.spinner("🔄 Downloading model..."):
        r = requests.get(MODEL_URL)
        with open(MODEL_PATH, "wb") as f:
            f.write(r.content)

@st.cache_resource
def load_model():
    model = models.resnet18(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, len(CLASS_NAMES))
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model

model = load_model()

# Image transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Top banner image
banner = Image.open("Screenshot 2025-05-08 203248.png")  # use your filename here
st.image(banner, use_column_width=True)

# Title section
st.markdown("""
    <h2 style='text-align: center; color: #2E2E2E;'>Welcome to <span style='color:#BF1363;'>PCOS Companion</span></h2>
    <p style='text-align: center; font-size: 18px;'>A Women's Best Friend – Predict PCOS from ultrasound images with AI</p>
""", unsafe_allow_html=True)

# Upload section
st.markdown("### 📤 Upload an Ultrasound Image")
uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])

if uploaded_file:
    try:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="🖼 Uploaded Ultrasound", use_column_width=True)

        with st.spinner("🔍 Analyzing..."):
            input_tensor = transform(image).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                output = model(input_tensor)
                _, predicted = torch.max(output, 1)
                confidence = torch.nn.functional.softmax(output, dim=1)[0][predicted.item()].item()
                prediction = CLASS_NAMES[predicted.item()]

        # Results
        st.success(f"💡 **Prediction:** {prediction}")
        st.info(f"📊 **Confidence Score:** {confidence * 100:.2f}%")

    except Exception as e:
        st.error(f"⚠️ Error: {str(e)}")

# Optional Footer
st.markdown("""
<hr>
<p style='text-align: center; font-size: 14px; color: grey;'>Empowering women's health through AI | © 2025 PCOS Companion</p>
""", unsafe_allow_html=True)
