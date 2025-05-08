import streamlit as st
from PIL import Image

# Set page layout
st.set_page_config(layout="wide")

# Apply custom CSS to make image full height and center text
st.markdown(
    """
    <style>
    .container {
        display: flex;
        height: 100vh;
    }
    .left-side {
        flex: 1;
        display: flex;
        justify-content: center;
        align-items: center;
        overflow: hidden;
        background-color: #f3f4f6;
    }
    .left-side img {
        width: 100%;
        height: 100%;
        object-fit: cover;
    }
    .right-side {
        flex: 1;
        display: flex;
        justify-content: center;
        align-items: center;
        background-color: #f3f4f6;
    }
    .text-content {
        max-width: 500px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Load the image
image = Image.open("4c4b7811-bbba-4123-8b31-c5d836ef62db.png")  # Use your image file name

# Display layout
st.markdown('<div class="container">', unsafe_allow_html=True)

# Left side: Image
st.markdown('<div class="left-side">', unsafe_allow_html=True)
st.image(image, use_column_width=True)
st.markdown('</div>', unsafe_allow_html=True)

# Right side: Text
st.markdown('<div class="right-side"><div class="text-content">', unsafe_allow_html=True)
st.markdown("### 👋 Welcome to **PCOS Detector**")
st.markdown("Please upload an ultrasound image to detect signs of Polycystic Ovary Syndrome (PCOS).")
st.markdown('</div></div>', unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)
