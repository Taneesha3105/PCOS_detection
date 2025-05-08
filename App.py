import streamlit as st
from PIL import Image
import torch
import torch.nn as nn
from torchvision import models, transforms
import os
import requests
import base64
from io import BytesIO

# Configuration
MODEL_URL = "https://github.com/Taneesha3105/PCOS_detection/releases/download/v1.0.0/PCOS_resnet18_model.pth"
MODEL_PATH = "PCOS_resnet18_model.pth"
CLASS_NAMES = ['No PCOS', 'PCOS']
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Streamlit config
st.set_page_config(page_title="PCOS Detector", page_icon="🧬")

# Convert image to base64
def get_image_base64(image_path):
    with open(image_path, "rb") as img_file:
        encoded = base64.b64encode(img_file.read()).decode()
    return encoded

# Load banner image
banner_path = "Screenshot 2025-05-08 203248.png"
image_base64 = get_image_base64(banner_path) if os.path.exists(banner_path) else ""

# Custom layout styling and structure
st.markdown(f"""
    <style>
    body, .stApp {{
        background-color: #f0f2f6;
        margin: 0;
        padding: 0;
    }}
    .main-container {{
        display: flex;
        height: 90vh;
        border-radius: 12px;
        overflow: hidden;
    }}
    .left-pane {{
        flex: 1;
        display: flex;
        align-items: center;
        justify-content: center;
        background-color: white;
    }}
    .left-pane img {{
        max-height: 100%;
        max-width: 100%;
        object-fit: contain;
    }}
    .right-pane {{
        flex: 1;
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
        padding: 2rem;
        background-color: #f0f2f6;
    }}
    </style>

    <div class="main-container">
        <div class="left-pane">
            <img src="data:image/png;base64,{image_base64}" alt="PCOS Banner" />
        </div>
        <div class="right-pane">
            <h2>👋 Welcome to <b>PCOS Detector</b></h2>
            <p><b>Please upload an ultrasound image to detect signs of Polycystic Ovary Syndrome (PCOS).</b></p>
        </div>
    </div>
""", unsafe_allow_html=True)

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

# Image pre-processing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Hide drag-and-drop text
st.markdown("""
    <style>
    div[data-testid="stFileUploader"] > label > div {
        display: none;
    }
    </style>
""", unsafe_allow_html=True)

# File upload
st.markdown("### 📤 Upload Ultrasound Image")
uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="📷 Uploaded Image", use_container_width=True)

        # Prediction
        with st.spinner("🔍 Analyzing image..."):
            input_tensor = transform(image).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                output = model(input_tensor)
                _, predicted = torch.max(output, 1)
                confidence = torch.nn.functional.softmax(output, dim=1)[0][predicted.item()].item()
                prediction = CLASS_NAMES[predicted.item()]

        st.success(f"🧠 **Prediction:** {prediction}")
        st.info(f"📊 **Confidence:** {confidence * 100:.2f}%")

    except Exception:
        st.error("⚠️ Invalid image file. Please try again.")
