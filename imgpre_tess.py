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

def preprocess_image(image: Image.Image, h, clahe_clip, clahe_tile, block_size, C, kernel_size, morph_iter):
    # Convert PIL Image to OpenCV format
    img = np.array(image)
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # CLAHE (adaptive histogram equalization)
    clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=(clahe_tile, clahe_tile))
    img = clahe.apply(img)
    # Denoising
    img = cv2.fastNlMeansDenoising(img, None, h=h, templateWindowSize=7, searchWindowSize=21)
    # Adaptive thresholding
    img = cv2.adaptiveThreshold(
        img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, block_size, C
    )
    # Morphological opening to remove small noise
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel, iterations=morph_iter)
    return img

def main():
    st.title("Myanmar NRC Image OCR (Interactive Preprocessing)")
    st.write("Upload an NRC image. Tune preprocessing parameters in real time to optimize OCR results.")

    uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg", "bmp", "tiff"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.subheader("Original Image")
        st.image(image, use_column_width=True)

        st.sidebar.header("Preprocessing Parameters")
        h = st.sidebar.slider("Denoising strength (h)", 5, 40, 19)
        clahe_clip = st.sidebar.slider("CLAHE clipLimit", 1, 10, 1)
        clahe_tile = st.sidebar.slider("CLAHE tileGridSize", 4, 32, 19)
        block_size = st.sidebar.slider("Adaptive Threshold Block Size", 11, 51, 33, step=2)
        C = st.sidebar.slider("Adaptive Threshold C", 0, 20, 20)
        kernel_size = st.sidebar.slider("Morph Kernel Size", 1, 5, 2)
        morph_iter = st.sidebar.slider("Morph Iterations", 1, 3, 1)

        # Option to test OCR on original image
        test_original = st.sidebar.checkbox("Show OCR result for original image", value=True)

        # Preprocess
        processed_img = preprocess_image(
            image, h, clahe_clip, clahe_tile, block_size, C, kernel_size, morph_iter
        )
        st.subheader("Processed Image (Tune parameters in sidebar)")
        st.image(processed_img, use_column_width=True, channels="GRAY")

        # OCR on processed image
        ocr_text = pytesseract.image_to_string(processed_img, config=CUSTOM_CONFIG)
        ocr_text = ocr_text.replace('|', 'I').replace('၀', '0').replace('သ်', 'ာ')
        ocr_text = ocr_text.replace('\n\n', '\n').strip()
        st.subheader("Extracted Myanmar Text (Processed Image)")
        st.text_area("OCR Result (Processed)", ocr_text, height=200)

        # OCR on original image (optional)
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
