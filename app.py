import streamlit as st
import cv2
from PIL import Image
import numpy as np

# Load the pre-trained Haar cascade model for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def detect_faces(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)
    return image

def main():
    # Page title with centered layout
    st.title("Face Detection App")
    st.markdown(
        """
        <style>
        .title {
            text-align: center;
            font-size: 36px;
            color: #34495e;
            padding: 20px;
            background-color: #ecf0f1;
            border-radius: 10px;
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        }
        .upload-section {
            background-color: #f8f9fa;
            padding: 20px;
            margin-top: 20px;
            border-radius: 10px;
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        }
        .result-section {
            background-color: #f8f9fa;
            padding: 20px;
            margin-top: 20px;
            border-radius: 10px;
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    
    st.markdown('<div class="title">Face Detection App</div>', unsafe_allow_html=True)
    
    # Main content sections
    st.markdown('<div class="upload-section">', unsafe_allow_html=True)
    st.write("Upload an image to detect faces.")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Convert the file to an opencv image.
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, 1)
        
        st.image(image, caption='Uploaded Image.', use_column_width=True)
        st.write("")
        st.write("Detecting faces...")
        
        result_image = detect_faces(image)
        
        # Convert the image to PIL format to display it with streamlit
        result_image = Image.fromarray(result_image)
        
        st.image(result_image, caption='Processed Image.', use_column_width=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
