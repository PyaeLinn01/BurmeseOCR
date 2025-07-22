"""
Streamlit App: NRC OCR using PaddleOCR for Myanmar Language
------------------------------------------------------------
Installation Guide:
    1. Install dependencies:
        pip install streamlit paddleocr paddlepaddle

Run the app:
    streamlit run paddle.py
"""

import streamlit as st
from paddleocr import PaddleOCR
from PIL import Image
import tempfile

# Initialize OCR globally
# Use English plus multilingual model which includes Myanmar text in PP-OCRv3/v2
ocr = PaddleOCR(use_angle_cls=True, rec=True, det=True, rec_algorithm='SVTR_LCNet')

st.title("Myanmar NRC OCR")
st.write("Upload an NRC image to extract Myanmar text.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Save image temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
        image.save(temp_file.name)
        temp_path = temp_file.name

    # Run OCR
    st.write("Extracting text...")
    result = ocr.ocr(temp_path)

    # Display extracted text
    if result and len(result[0]) > 0:
        st.subheader("Extracted Myanmar Text:")
        for idx, line in enumerate(result[0], start=1):
            st.write(f"{idx}: {line[1][0]}")
    else:
        st.write("No text detected.")