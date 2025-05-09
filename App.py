import streamlit as st
from PIL import Image
import torch
import torch.nn as nn
from torchvision import models, transforms
import os
import requests
import google.generativeai as genai

# ==== GEMINI CONFIGURATION ====
GOOGLE_API_KEY = "AIzaSyBZqGn9XXw8ML1uUHaqjulYOGwyHhfa2as"
genai.configure(api_key=GOOGLE_API_KEY)
chat_model = genai.GenerativeModel(model_name="models/gemini-2.0-flash")
chat_session = chat_model.start_chat()

# ==== MODEL CONFIGURATION ====
MODEL_URL = "https://github.com/Taneesha3105/PCOS_detection/releases/download/v1.0.0/PCOS_resnet18_model.pth"
MODEL_PATH = "PCOS_resnet18_model.pth"
CLASS_NAMES = ['PCOS', 'No PCOS']
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

st.set_page_config(page_title="PCOSense", layout="wide", page_icon="🧬")
# ==== CUSTOM CSS FOR FONT ====
st.markdown("""
    <style>
    html, body, [class*="css"] {
        font-family: 'Century Gothic', sans-serif;
        font-weight: bold;
    }
    .big-font {
        font-size: 42px !important;
        font-weight: bold !important;
    }
    .medium-font {
        font-size: 22px !important;
        font-weight: bold !important;
    }
    .small-font {
        font-size: 18px !important;
        font-weight: bold !important;
    }
    </style>
""", unsafe_allow_html=True)



# ==== HEADER ====
st.markdown('<div class="big-font">🧬PCOSense</div>', unsafe_allow_html=True)

st.markdown('<div class="medium-font">We aim to simplify the process of PCOS detection in females.</div>', unsafe_allow_html=True)
st.markdown('<div class="small-font">Upload an ultrasound image below to detect signs of Polycystic Ovary Syndrome (PCOS) using Machine Learning.</div>', unsafe_allow_html=True)
st.markdown('<div class="small-font">You can also ask health-related questions using our Gemini Chat Assistant.</div>', unsafe_allow_html=True)
st.markdown('<div class="small-font">Thank you for using our app!😊</div>', unsafe_allow_html=True)


# ==== BANNER IMAGE ====
banner_path = "ChatGPT Image May 9, 2025, 10_39_26 AM.png"  # Update this if needed
if os.path.exists(banner_path):
    st.image(banner_path, use_container_width=True)

# ==== DOWNLOAD MODEL ====
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

# ==== TRANSFORMS ====
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# ==== HIDE LABEL ====
st.markdown("""
    <style>
    div[data-testid="stFileUploader"] > label > div {
        display: none;
    }
    </style>
""", unsafe_allow_html=True)

# ==== IMAGE UPLOAD ====
uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="📷 Uploaded Image", use_container_width=True)

        with st.spinner("🔍 Analyzing image..."):
            input_tensor = transform(image).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                output = model(input_tensor)
                _, predicted = torch.max(output, 1)
                confidence = torch.nn.functional.softmax(output, dim=1)[0][predicted.item()].item()
                prediction = CLASS_NAMES[predicted.item()]

        st.success(f"🧠 *Prediction:* {prediction}")
        st.info(f"📊 *Confidence:* {confidence * 100:.2f}%")

    except Exception as e:
        st.error("⚠ Invalid image file. Please try again.")

# ==== GEMINI CHAT ====
st.markdown("---")
st.markdown('<div class="medium-font">🤖 Gemini Chat Assistant</div>', unsafe_allow_html=True)
st.markdown('<div class="small-font">Ask anything about PCOS, ultrasound diagnostics, or how this app works!</div>', unsafe_allow_html=True)

user_input = st.text_input("💬 Please ask a question:")
if user_input:
    with st.spinner("Gemini is thinking..."):
        response = chat_session.send_message(user_input)
        st.markdown(f"🧠 Chatbot: **{response.text}**")
