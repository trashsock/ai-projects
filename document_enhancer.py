import streamlit as st
import cv2
import numpy as np
from PIL import Image
import pytesseract
import matplotlib.pyplot as plt

# Image enhancement function
def enhance_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    denoised = cv2.fastNlMeansDenoising(gray, None, 30, 7, 21)
    enhanced = cv2.adaptiveThreshold(denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                      cv2.THRESH_BINARY, 11, 2)
    return enhanced

# OCR function
def extract_text(image):
    return pytesseract.image_to_string(image, lang='eng')

# App interface
st.title("📄 Document Enhancement Tool")

st.write("Upload a scanned or low-quality document image, and the tool will enhance its readability and extract text using OCR.")

# File uploader
uploaded_file = st.file_uploader("Upload an image file", type=["png", "jpg", "jpeg"])

if uploaded_file:
    # Read and process the image
    image = np.array(Image.open(uploaded_file))
    st.image(image, caption="Uploaded Image", use_column_width=True)

    st.write("Enhancing the image...")
    enhanced_image = enhance_image(image)
    st.image(enhanced_image, caption="Enhanced Image", use_column_width=True, channels="GRAY")

    st.write("Extracting text...")
    extracted_text = extract_text(enhanced_image)
    st.text_area("Extracted Text", extracted_text, height=200)

    # Save enhanced image
    save_button = st.button("Download Enhanced Image")
    if save_button:
        enhanced_pil = Image.fromarray(enhanced_image)
        enhanced_pil.save("enhanced_image.png")
        st.success("Enhanced image saved as 'enhanced_image.png'.")

