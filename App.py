# Inject custom CSS for background, font style, and size
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600&display=swap');

    html, body, [class*="css"]  {
        font-family: 'Poppins', sans-serif !important;
        font-size: 18px !important;
        background-color: #f0f0f0;
    }

    h1, h2, h3, h4, h5, h6 {
        font-weight: 600 !important;
        color: #222;
    }

    .stMarkdown {
        font-size: 18px !important;
    }

    .stButton > button {
        font-size: 18px !important;
    }

    .stTextInput > div > input {
        font-size: 18px !important;
    }

    .stSelectbox > div > div {
        font-size: 18px !important;
    }

    .stFileUploader {
        font-size: 18px !important;
    }
    </style>
""", unsafe_allow_html=True)
