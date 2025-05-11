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
chat_session = chat_model.start_chat(history=[
    {
        "role": "system", 
        "parts": ["You are a helpful PCOS assistant. Provide empathetic, accurate information about Polycystic Ovary Syndrome (PCOS), its symptoms, treatments, and management strategies. Do not provide medical diagnosis."]
    }
])

# ==== MODEL CONFIGURATION ====
MODEL_URL = "https://github.com/Taneesha3105/PCOS_detection/releases/download/v1.0.0/PCOS_resnet18_model.pth"
MODEL_PATH = "PCOS_resnet18_model.pth"
CLASS_NAMES = ['PCOS', 'No PCOS']
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==== PAGE CONFIGURATION ====
st.set_page_config(
    page_title="PCOSense Companion", 
    layout="wide", 
    page_icon="🌸",
    initial_sidebar_state="expanded"
)

# ==== CUSTOM CSS ====
st.markdown("""
    <style>
    .main {
        background-color: #f8f1f4;
    }
    .stApp {
        font-family: 'Arial', sans-serif;
    }
    .big-font {
        font-size: 42px !important;
        font-weight: 700;
        color: #d94c63;
    }
    .medium-font {
        font-size: 22px !important;
        color: #444;
    }
    .small-font {
        font-size: 16px !important;
        color: #666;
    }
    .stButton>button {
        background-color: #d94c63;
        color: white;
        border-radius: 20px;
        padding: 10px 25px;
        border: none;
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: #c03651;
    }
    .info-box {
        background-color: #fff;
        border-radius: 10px;
        padding: 20px;
        margin: 10px 0;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
    }
    .prediction-box {
        padding: 20px;
        border-radius: 10px;
        margin-top: 20px;
        text-align: center;
    }
    .pcos-positive {
        background-color: rgba(217, 76, 99, 0.1);
        border: 1px solid #d94c63;
    }
    .pcos-negative {
        background-color: rgba(75, 192, 192, 0.1);
        border: 1px solid #4bc0c0;
    }
    .tabs-font {
        font-size: 18px !important;
        font-weight: bold;
    }
    div[data-testid="stFileUploader"] > label > div {
        display: none;
    }
    </style>
""", unsafe_allow_html=True)

# ==== SIDEBAR ====
with st.sidebar:
    st.image("https://raw.githubusercontent.com/yourusername/PCOSense/main/logo.png", width=100)
    st.markdown('<div class="medium-font">PCOSense Companion</div>', unsafe_allow_html=True)
    st.markdown("---")
    st.markdown('<div class="small-font">A women\'s best friend for PCOS detection and support</div>', unsafe_allow_html=True)
    st.markdown("---")
    
    st.markdown("### Quick Facts")
    st.info("• PCOS affects about 1 in 5 (20%) Indian women")
    st.info("• It affects 5% to 10% of women in their reproductive age")
    st.info("• PCOS is a leading cause of female infertility")
    st.info("• Early diagnosis can help manage symptoms effectively")
    
    st.markdown("---")
    st.markdown("### Resources")
    st.markdown("📚 [PCOS Diet Guide](https://example.com)")
    st.markdown("🧘‍♀️ [Exercise Recommendations](https://example.com)")
    st.markdown("👩‍⚕️ [Find a Specialist](https://example.com)")

# ==== MAIN PAGE ====
st.markdown('<div class="big-font">🌸 PCOSense Companion</div>', unsafe_allow_html=True)
st.markdown('<div class="medium-font">AI-powered PCOS detection and support system</div>', unsafe_allow_html=True)

# ==== TABS ====
tab1, tab2, tab3 = st.tabs(["🔍 PCOS Detection", "❓ About PCOS", "💬 Ask An Expert"])

with tab1:
    st.markdown('<div class="small-font">Upload an ultrasound image to detect signs of Polycystic Ovary Syndrome (PCOS)</div>', unsafe_allow_html=True)
    
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

    # ==== IMAGE UPLOAD ====
    uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png"])

    col1, col2 = st.columns([1, 1])
    
    if uploaded_file is not None:
        try:
            image = Image.open(uploaded_file).convert("RGB")
            with col1:
                st.image(image, caption="📷 Uploaded Ultrasound Image", use_container_width=True)

            with col2:
                with st.spinner("🔍 Analyzing image..."):
                    input_tensor = transform(image).unsqueeze(0).to(DEVICE)
                    with torch.no_grad():
                        output = model(input_tensor)
                        _, predicted = torch.max(output, 1)
                        confidence = torch.nn.functional.softmax(output, dim=1)[0][predicted.item()].item()
                        prediction = CLASS_NAMES[predicted.item()]
                
                if prediction == "PCOS":
                    st.markdown(f"""
                    <div class="prediction-box pcos-positive">
                        <h2>🔍 Result: PCOS Detected</h2>
                        <p>Confidence: {confidence * 100:.2f}%</p>
                        <p>This ultrasound image shows potential signs of Polycystic Ovary Syndrome.</p>
                        <p><b>Important:</b> This is not a medical diagnosis. Please consult a healthcare professional.</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="prediction-box pcos-negative">
                        <h2>🔍 Result: No PCOS Detected</h2>
                        <p>Confidence: {confidence * 100:.2f}%</p>
                        <p>This ultrasound image does not show typical signs of Polycystic Ovary Syndrome.</p>
                        <p><b>Note:</b> Always consult with healthcare professionals for proper diagnosis.</p>
                    </div>
                    """, unsafe_allow_html=True)

        except Exception as e:
            st.error("⚠ Invalid image file. Please try again.")
    
    else:
        st.markdown("""
        <div class="info-box">
            <h3>How to use the PCOS Detection tool:</h3>
            <ol>
                <li>Upload a clear ultrasound image of the ovaries</li>
                <li>Our AI model will analyze the image</li>
                <li>Results will appear showing whether PCOS indicators are detected</li>
                <li>Remember that this tool is for educational purposes only</li>
            </ol>
        </div>
        """, unsafe_allow_html=True)

with tab2:
    st.markdown('<div class="medium-font">Understanding Polycystic Ovary Syndrome</div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-box">
        <h3>What is PCOS?</h3>
        <p>Polycystic Ovary Syndrome, or PCOS, is a health condition that affects about one in five (20%) Indian women. It affects 5% to 10% of women in their reproductive age and is a leading cause of female infertility.</p>
        
        <h3>Common Symptoms</h3>
        <ul>
            <li>Irregular periods or no periods</li>
            <li>Difficulty getting pregnant</li>
            <li>Excessive hair growth (hirsutism)</li>
            <li>Weight gain</li>
            <li>Thinning hair and hair loss from the head</li>
            <li>Oily skin or acne</li>
        </ul>
        
        <h3>Risk Factors</h3>
        <ul>
            <li>Family history of PCOS</li>
            <li>Obesity</li>
            <li>Insulin resistance</li>
        </ul>
        
        <h3>Management Strategies</h3>
        <ul>
            <li>Healthy diet and regular exercise</li>
            <li>Medications to regulate periods</li>
            <li>Treatments for specific symptoms</li>
            <li>Regular monitoring by healthcare professionals</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="info-box">
            <h3>Diet & Nutrition</h3>
            <p>A balanced diet can help manage PCOS symptoms:</p>
            <ul>
                <li>Focus on low-glycemic foods</li>
                <li>Increase fiber intake</li>
                <li>Include anti-inflammatory foods</li>
                <li>Stay hydrated</li>
                <li>Limit processed foods and added sugars</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="info-box">
            <h3>Exercise & Lifestyle</h3>
            <p>Regular physical activity can help with PCOS management:</p>
            <ul>
                <li>Aim for 150 minutes of moderate exercise weekly</li>
                <li>Include both cardio and strength training</li>
                <li>Practice stress-reducing activities like yoga</li>
                <li>Ensure adequate sleep</li>
                <li>Avoid smoking and limit alcohol</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

with tab3:
    st.markdown('<div class="medium-font">Ask Our AI Assistant About PCOS</div>', unsafe_allow_html=True)
    st.markdown('<div class="small-font">Get answers to your questions about PCOS symptoms, management, and more.</div>', unsafe_allow_html=True)
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # User input
    if prompt := st.chat_input("Ask anything about PCOS..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate a response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = chat_session.send_message(prompt)
                st.markdown(response.text)
        
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response.text})

# ==== FOOTER ====
st.markdown("---")
footer_col1, footer_col2, footer_col3 = st.columns([1, 2, 1])

with footer_col1:
    st.markdown("### Connect With Us")
    st.markdown("📱 Follow on [Instagram](https://instagram.com)")
    st.markdown("🐦 Follow on [Twitter](https://twitter.com)")
    st.markdown("📘 Join our [Facebook](https://facebook.com) group")

with footer_col2:
    st.markdown("### Disclaimer")
    st.markdown("This application is for educational purposes only and is not intended to provide personal medical advice. Always consult qualified healthcare providers for diagnosis and treatment of any medical condition.")

with footer_col3:
    st.markdown("### Support PCOSense")
    st.markdown("❤️ [Donate to our cause](https://example.com)")
    st.markdown("🤝 [Volunteer opportunities](https://example.com)")
    st.markdown("📧 [Contact us](mailto:info@pcosense.org)")
