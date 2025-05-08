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
st.set_page_config(page_title="PCOS Companion", page_icon="🧬")

st.markdown("""
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    div[data-testid="stFileUploader"] > label > div {display: none;}
    .centered {
        text-align: center;
        margin-top: 20px;
    }
    </style>
""", unsafe_allow_html=True)

# ---------------------------- Hero Banner ----------------------------
st.image("68c4c55d-9cef-4d8e-8a29-658025b7a6fa.png", use_column_width=True)

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

# ---------------------------- Chatbot UI ----------------------------
st.markdown("## 💬 Chat with PCOS Assistant")

faq_responses = {
    "whats pcos?": "Polycystic ovary syndrome (PCOS) is a common hormonal disorder in women of reproductive age, often characterized by irregular periods, excess androgen levels, and cysts on the ovaries. It can also lead to other health issues like insulin resistance and an increased risk of type 2 diabetes. While there's no cure, symptoms can be managed through lifestyle changes, medication, and in some cases, surgery.",
    "how to use the app?": "Just upload the ultrasound image of the ovary of the patient and you are good to go! Our app detects if you have PCOS or not",
    "i face errors using your app": "Sorry for the inconvenience, please ask our help desk!"
}

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Welcome to the PCOS Companion chatbot. How can I assist you today?"}]

user_input = st.text_input("Ask a question:")

if user_input:
    user_query = user_input.strip().lower()
    response = faq_responses.get(user_query, "Sorry, I don't have an answer for that right now.")

    st.session_state.messages.append({"role": "user", "content": user_input})
    st.session_state.messages.append({"role": "assistant", "content": response})

for i, msg in enumerate(st.session_state.messages):
    message(msg["content"], is_user=(msg["role"] == "user"), key=f"chat_{i}")
