import streamlit as st
from PIL import Image
import torch
import torch.nn as nn
from torchvision import models, transforms
import os
import requests

# Configuration

MODEL\_URL = "[https://github.com/Taneesha3105/PCOS\_detection/releases/download/v1.0.0/PCOS\_resnet18\_model.pth](https://github.com/Taneesha3105/PCOS_detection/releases/download/v1.0.0/PCOS_resnet18_model.pth)"
MODEL\_PATH = "PCOS\_resnet18\_model.pth"
CLASS\_NAMES = \['No PCOS', 'PCOS']
DEVICE = torch.device("cuda" if torch.cuda.is\_available() else "cpu")

# Streamlit config

st.set\_page\_config(page\_title="PCOS Detector", page\_icon="🧬")

# Display side-by-side layout

col1, col2 = st.columns(\[1, 1])

with col1:
banner\_path = "Screenshot 2025-05-08 203248.png"
if os.path.exists(banner\_path):
st.image(banner\_path, use\_container\_width=True)

with col2:
st.markdown("## 👋WELCOME TO THE PCOS DETECTOR!")
st.markdown("We aim to simplify the process of PCOS detection in females")
st.markdown("Please upload an ultrasound image below to detect signs of Polycystic Ovary Syndrome (PCOS)")

# Download model if not present

if not os.path.exists(MODEL\_PATH):
with st.spinner("🔄 Downloading model..."):
r = requests.get(MODEL\_URL)
with open(MODEL\_PATH, "wb") as f:
f.write(r.content)

@st.cache\_resource
def load\_model():
model = models.resnet18(pretrained=False)
model.fc = nn.Linear(model.fc.in\_features, len(CLASS\_NAMES))
model.load\_state\_dict(torch.load(MODEL\_PATH, map\_location=DEVICE))
model.to(DEVICE)
model.eval()
return model

model = load\_model()

# Image pre-processing

transform = transforms.Compose(\[
transforms.Resize((224, 224)),
transforms.ToTensor(),
transforms.Normalize(mean=\[0.485, 0.456, 0.406],
std=\[0.229, 0.224, 0.225])
])

# Hide drag-and-drop text

st.markdown(""" <style>
div\[data-testid="stFileUploader"] > label > div {
display: none;
} </style>
""", unsafe\_allow\_html=True)

# File upload

uploaded\_file = st.file\_uploader("", type=\["jpg", "jpeg", "png"])

if uploaded\_file is not None:
try:
image = Image.open(uploaded\_file).convert("RGB")
st.image(image, caption="📷 Uploaded Image", use\_container\_width=True)
\# Prediction
with st.spinner("🔍 Analyzing image..."):
input\_tensor = transform(image).unsqueeze(0).to(DEVICE)
with torch.no\_grad():
output = model(input\_tensor)
\_, predicted = torch.max(output, 1)
confidence = torch.nn.functional.softmax(output, dim=1)\[0]\[predicted.item()].item()
prediction = CLASS\_NAMES\[predicted.item()]

st.success(f"🧠 **Prediction:** {prediction}")
st.info(f"📊 **Confidence:** {confidence * 100:.2f}%")


except Exception:
st.error("⚠️ Invalid image file. Please try again.")
