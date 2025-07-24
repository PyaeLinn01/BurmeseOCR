import streamlit as st
import numpy as np
import cv2
from PIL import Image
import pytesseract
import os
import subprocess
import json

# Tesseract config
os.environ['TESSDATA_PREFIX'] = '/usr/share/tesseract-ocr/5/tessdata'
CUSTOM_CONFIG = r'--oem 3 --psm 6 -l mya'

def preprocess_image(image: Image.Image):
    img = np.array(image)
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Increase contrast
    img = cv2.equalizeHist(img)
    # Adaptive Threshold (binarization)
    img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                cv2.THRESH_BINARY_INV, 31, 15)
    # Remove small noise
    kernel = np.ones((2, 2), np.uint8)
    img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel, iterations=1)
    # Resize for better OCR
    img = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    return img

def correct_text_with_ollama(text):
    # Send text to local Ollama model
    prompt = f"Correct and complete this Myanmar NRC text: {text}"
    cmd = ["ollama", "run", "llama3", "--prompt", prompt]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.stdout.strip()

def main():
    st.title("Myanmar NRC Handwritten OCR + LLM Correction")
    uploaded_file = st.file_uploader("Upload NRC Image", type=["png", "jpg", "jpeg"])
    
    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        processed_img = preprocess_image(image)
        st.subheader("Processed Image")
        st.image(processed_img, channels="GRAY")

        # OCR with Tesseract
        raw_text = pytesseract.image_to_string(processed_img, config=CUSTOM_CONFIG)
        raw_text = raw_text.replace('|', 'I').replace('၀', '0').strip()
        st.subheader("Raw OCR Output")
        st.text_area("Tesseract Result", raw_text, height=200)

        # LLM correction (Ollama)
        if st.button("Correct with Ollama"):
            corrected_text = correct_text_with_ollama(raw_text)
            st.subheader("Corrected NRC Text")
            st.text_area("LLM Output", corrected_text, height=200)

if __name__ == "__main__":
    main()