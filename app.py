import os
import subprocess

# 🚀 THE MAGIC FIX: Force uninstall the GUI version of OpenCV that crashes Streamlit
subprocess.call(['pip', 'uninstall', '-y', 'opencv-python'])

import streamlit as st
from ultralytics import YOLO
import PIL.Image
import numpy as np

# Page Configuration
st.set_page_config(page_title="Optical Surface Inspector", layout="wide")

# Load the Model from the 'models' directory
@st.cache_resource
def load_model():
    return YOLO("models/best.onnx", task="detect")

model = load_model()

# Header Section
st.title("🔍 Automated Optical Surface Inspector")
st.write("Upload an image of the lens or glass slide to analyze its topography and detect defects.")

# Sidebar for Image Upload
uploaded_file = st.sidebar.file_uploader("Choose an image to analyze...", type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    # Read the uploaded image
    image = PIL.Image.open(uploaded_file)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Original Image")
        st.image(image, use_container_width=True)
        
    with col2:
        st.subheader("Smart Analysis Results")
        # Run inference
        results = model.predict(image, conf=0.25)
        res_plotted = results[0].plot()
        
        # Display the image with bounding boxes
        st.image(res_plotted, caption='Detected Defects', use_container_width=True)

    # Analytics Section
    st.divider()
    st.header("📊 Technical Report")
    
    # Calculate metrics
    defect_count = len(results[0].boxes)
    avg_conf = np.mean(results[0].boxes.conf.cpu().numpy()) if defect_count > 0 else 0
    
    metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
    
    with metrics_col1:
        status = "❌ Defective" if defect_count > 0 else "✅ Regular (Perfect)"
        st.metric("Surface Status", status)
        
    with metrics_col2:
        st.metric("Detected Defects Count", defect_count)
        
    with metrics_col3:
        st.metric("Average Model Confidence", f"{avg_conf:.2%}")

    # Final Recommendation
    if defect_count > 0:
        st.warning("⚠️ Defects detected on the surface topography. It is recommended to review the item before use.")
    else:
        st.success("✨ The surface meets technical specifications and optical regularity.")
