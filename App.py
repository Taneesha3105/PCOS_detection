import streamlit as st
from PIL import Image

# Set page layout
st.set_page_config(layout="wide")

# Apply custom CSS for side-by-side layout
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
        background-color: #f3f4f6;
        overflow: hidden;
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
        text-align: center;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Create the layout container
st.markdown('<div class="container">', unsafe_allow_html=True)

# Left side: Uploaded image
st.markdown('<div class="left-side">', unsafe_allow_html=True)
uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, use_column_width=True)
else:
    # Display default image if no upload yet
    default_image = Image.open("default-image.png")  # Make sure this image exists in your repo
    st.image(default_image, use_column_width=True)
st.markdown('</div>', unsafe_allow_html=True)

# Right side: Text
st.markdown('<div class="right-side"><div class="text-content">', unsafe_allow_html=True)
st.markdown("### 👋 Welcome to **PCOS Detector**")
st.markdown("Please upload an ultrasound image to detect signs of **Polycystic Ovary Syndrome (PCOS)**.")
st.markdown('</div></div>', unsafe_allow_html=True)

# Close container
st.markdown('</div>', unsafe_allow_html=True)
