import streamlit as st
from PIL import Image

# Page setup
st.set_page_config(page_title="PCOS Detector", layout="wide")

# Title and description area
st.markdown("## 👋 Welcome to **PCOS Detector**")
st.markdown("Please upload an ultrasound image to detect signs of **Polycystic Ovary Syndrome (PCOS)**.")

# Divide the page into two columns
col1, col2 = st.columns(2)

with col1:
    uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
    else:
        # Fallback image if no upload — change this filename to your image
        try:
            default_image = Image.open("4c4b7811-bbba-4123-8b31-c5d836ef62db.png")  # or use a file you know exists
            st.image(default_image, caption="Default Illustration", use_column_width=True)
        except FileNotFoundError:
            st.warning("Please upload an image.")

with col2:
    st.markdown("### 📌 Instructions")
    st.write(
        """
        - Upload a clear ultrasound image.
        - The model will analyze the image and check for PCOS signs.
        - Results will appear below once implemented.
        """
    )
