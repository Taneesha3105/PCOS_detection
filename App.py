import streamlit as st
from PIL import Image
import torch
import torch.nn as nn
from torchvision import models, transforms
import os
import requests
import google.generativeai as genai

# === CONFIG ===
st.set_page_config(page_title="PCOSense", layout="wide", page_icon="🧬")

# === GEMINI SETUP ===
GOOGLE_API_KEY = "AIzaSyBZqGn9XXw8ML1uUHaqjulYOGwyHhfa2as"
genai.configure(api_key=GOOGLE_API_KEY)
chat_model = genai.GenerativeModel(model_name="models/gemini-2.0-flash")
chat_session = chat_model.start_chat()

# === MODEL SETUP ===
MODEL_URL = "https://github.com/Taneesha3105/PCOS_detection/releases/download/v1.0.0/PCOS_resnet18_model.pth"
MODEL_PATH = "PCOS_resnet18_model.pth"
CLASS_NAMES = ['PCOS', 'No PCOS']
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === STYLING ===
st.markdown("""
    <style>
        html, body, [class*="css"] {
            font-family: 'Segoe UI', sans-serif;
            background-color: #fafafa;
        }
        .title {
            font-size: 40px;
            font-weight: bold;
            color: #5f27cd;
        }
        .subtitle {
            font-size: 22px;
            margin-top: -10px;
            color: #333333;
        }
        .section {
            margin-top: 40px;
            padding: 30px 20px;
            background-color: #ffffff;
            border-radius: 12px;
            box-shadow: 0px 2px 10px rgba(0,0,0,0.05);
        }
    </style>
""", unsafe_allow_html=True)

# === HEADER ===
st.markdown('<div class="title">🧬 PCOSense</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">AI-powered tool for early detection of Polycystic Ovary Syndrome (PCOS)</div>', unsafe_allow_html=True)

# === MODEL DOWNLOADER ===
if not os.path.exists(MODEL_PATH):
    with st.spinner("Downloading model..."):
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

# === IMAGE SECTION ===
st.markdown('<div class="section">', unsafe_allow_html=True)
st.subheader("Upload an Ultrasound Image")
st.write("Our model analyzes ultrasound images to check for signs of PCOS. Supported formats: JPG, JPEG, PNG.")

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file).convert("RGB")
        col1, col2 = st.columns([1, 2])

        with col1:
            st.image(image, caption="📷 Uploaded Image", use_container_width=True)

        with col2:
            st.write("🔍 **Processing Image...**")
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])
            input_tensor = transform(image).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                output = model(input_tensor)
                _, predicted = torch.max(output, 1)
                confidence = torch.nn.functional.softmax(output, dim=1)[0][predicted.item()].item()
                prediction = CLASS_NAMES[predicted.item()]

            st.success(f"🧠 **Prediction:** {prediction}")
            st.info(f"📊 **Confidence:** {confidence * 100:.2f}%")

    except Exception as e:
        st.error("⚠️ Invalid image file. Please try again.")
st.markdown("</div>", unsafe_allow_html=True)

# === GEMINI SECTION ===
st.markdown('<div class="section">', unsafe_allow_html=True)
st.subheader("💬 Gemini Health Assistant")
st.write("Ask anything about PCOS, diagnostics, or how this app works.")

user_input = st.text_input("Type your question here:")
if user_input:
    with st.spinner("Gemini is thinking..."):
        response = chat_session.send_message(user_input)
        st.markdown(f"🧠 **Chatbot:** *{response.text}*")
st.markdown("</div>", unsafe_allow_html=True)
