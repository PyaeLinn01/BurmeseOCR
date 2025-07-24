# Requirements:
# streamlit, opencv-python, pytesseract, pillow

import streamlit as st
import numpy as np
import cv2
from PIL import Image
import pytesseract
import os

# Set Tesseract config for Myanmar only
os.environ['TESSDATA_PREFIX'] = '/usr/share/tesseract-ocr/5/tessdata'
CUSTOM_CONFIG = r'--oem 3 --psm 6 -l mya'

def preprocess_image(image: Image.Image):
    # Convert PIL Image to OpenCV format
    img = np.array(image)
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Denoising
    img = cv2.fastNlMeansDenoising(img, None, h=30, templateWindowSize=7, searchWindowSize=21)
    # Adaptive thresholding
    img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                cv2.THRESH_BINARY, 31, 11)
    return img

def main():
    st.title("Myanmar NRC Image OCR (with Advanced Preprocessing)")
    st.write("Upload an NRC image. The app will preprocess and extract Myanmar text only.")

    uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg", "bmp", "tiff"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.subheader("Original Image")
        st.image(image, use_column_width=True)

        # Preprocess
        processed_img = preprocess_image(image)
        st.subheader("Processed Image")
        st.image(processed_img, use_column_width=True, channels="GRAY")

        # OCR
        ocr_text = pytesseract.image_to_string(processed_img, config=CUSTOM_CONFIG)
        ocr_text = ocr_text.replace('|', 'I').replace('၀', '0').replace('သ်', 'ာ')
        ocr_text = ocr_text.replace('\n\n', '\n').strip()
        st.subheader("Extracted Myanmar Text")
        st.text_area("OCR Result", ocr_text, height=200)

        test_original = st.checkbox("Show OCR Result for Original Image")
        if test_original:
            st.subheader("Extracted Myanmar Text (Original Image)")
            orig_img = np.array(image)
            if orig_img.ndim == 3:
                orig_img = cv2.cvtColor(orig_img, cv2.COLOR_RGB2GRAY)
            orig_ocr_text = pytesseract.image_to_string(orig_img, config=CUSTOM_CONFIG)
            orig_ocr_text = orig_ocr_text.replace('|', 'I').replace('၀', '0').replace('သ်', 'ာ')
            orig_ocr_text = orig_ocr_text.replace('\n\n', '\n').strip()
            st.text_area("OCR Result (Original)", orig_ocr_text, height=200)

if __name__ == "__main__":
    main()
