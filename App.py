import streamlit as st
from PIL import Image
import torch
import torch.nn as nn
from torchvision import models, transforms
import os
import requests
from streamlit_chat import message

# ---------------------------- Configuration ----------------------------
MODEL_URL = "https://github.com/Taneesha3105/PCOS_detection/releases/download/v1.0.0/PCOS_resnet18_model.pth"
MODEL_PATH = "PCOS_resnet18_model.pth"
CLASS_NAMES = ['PCOS', 'No PCOS']
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------- Streamlit Page Setup ----------------------------
st.set_page_config(page_title="PCOS Predictor", page_icon="🧬")

st.markdown("""
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    div[data-testid="stFileUploader"] > label > div {display: none;}
    </style>
""", unsafe_allow_html=True)

# ---------------------------- Download Model ----------------------------
if not os.path.exists(MODEL_PATH):
    with st.spinner("🔄 Downloading model..."):
        r = requests.get(MODEL_URL)
        with open(MODEL_PATH, "wb") as f:
            f.write(r.content)

# ---------------------------- Load Model ----------------------------
@st.cache_resource
def load_model():
    model = models.resnet18(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, len(CLASS_NAMES))
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model

model = load_model()

# ---------------------------- Image Preprocessing ----------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# ---------------------------- Image Classification UI ----------------------------
st.title("🧬 PCOS Ultrasound Analyzer")
st.markdown("Upload an **ultrasound image** to detect signs of **Polycystic Ovary Syndrome (PCOS)** using AI.")

uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="📷 Uploaded Image", use_column_width=True)

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

# ---------------------------- PCOS Chatbot ----------------------------
st.markdown("## 💬 Chat with PCOS Assistant")

if "messages" not in st.session_state:
    st.session_state.messages = []

user_input = st.text_input("Ask a question about PCOS:")

if user_input:
    pcos_keywords = [
        "pcos", "polycystic", "ovary", "syndrome", "symptoms", "causes",
        "treatment", "diagnosis", "irregular periods", "hormonal", "fertility",
        "insulin", "androgen", "cysts", "ultrasound", "medicine", "weight", "hair"
    ]

    user_text = user_input.lower()
    if any(word in user_text for word in pcos_keywords):
        if "symptom" in user_text:
            response = "Common PCOS symptoms include irregular periods, excess androgen, acne, and ovarian cysts."
        elif "cause" in user_text:
            response = "PCOS may be caused by excess insulin, low-grade inflammation, or hereditary factors."
        elif "treatment" in user_text or "cure" in user_text:
            response = "PCOS treatments include lifestyle changes, hormonal therapy, and medications like metformin."
        elif "diagnosis" in user_text:
            response = "Doctors diagnose PCOS using blood tests, ultrasound imaging, and evaluation of symptoms."
        else:
            response = "PCOS is a hormonal disorder affecting women, often leading to irregular periods and cysts in ovaries."
    else:
        response = "Unrelated to PCOS."

    st.session_state.messages.append({"role": "user", "content": user_input})
    st.session_state.messages.append({"role": "assistant", "content": response})

# Display chat history
for msg in st.session_state.messages:
    message(msg["content"], is_user=(msg["role"] == "user"))
