import streamlit as st
import numpy as np
import cv2
from PIL import Image
import easyocr

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
    st.title("Myanmar Handwritten OCR with EasyOCR")
    st.write("Upload an NRC image with handwritten Myanmar text. The app will preprocess and extract text using EasyOCR.")

    uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg", "bmp", "tiff"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.subheader("Original Image")
        st.image(image, use_column_width=True)

        # Preprocess
        processed_img = preprocess_image(image)
        st.subheader("Processed Image (Handwriting Enhanced)")
        st.image(processed_img, use_column_width=True, channels="GRAY")

        # OCR with EasyOCR
        reader = easyocr.Reader(['en'], gpu=False)
        result = reader.readtext(processed_img, detail=0, paragraph=True)
        ocr_text = "\n".join(result)
        st.subheader("Extracted Myanmar Text (EasyOCR)")
        st.text_area("OCR Result", ocr_text, height=200)

if __name__ == "__main__":
    main()
