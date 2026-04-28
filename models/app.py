import streamlit as st
from ultralytics import YOLO
import PIL.Image
import numpy as np

# --- Page Configuration ---
st.set_page_config(
    page_title="Optical Surface Inspector", 
    page_icon="🔍", 
    layout="wide"
)

# --- Model Loading ---
# Using st.cache_resource ensures the 11.7 MB ONNX model loads only once
@st.cache_resource
def load_model():
    # Load the optimized ONNX model
    return YOLO("best.onnx", task="detect")

model = load_model()

# --- Main UI ---
st.title("🔍 Automated Optical Surface Inspector")
st.markdown("Upload a high-resolution image of a glass slide or lens. The AI will analyze the surface topography and map out irregularities.")

# --- Sidebar Controls ---
with st.sidebar:
    st.header("⚙️ Inspection Settings")
    # Dynamic confidence threshold (defaults to the 0.25 used in training testing)
    conf_threshold = st.slider(
        "Confidence Threshold", 
        min_value=0.05, 
        max_value=0.95, 
        value=0.25, 
        step=0.05,
        help="Lower values detect more potential defects. Higher values are stricter."
    )
    st.divider()
    st.markdown("**Supported Formats:** JPG, JPEG, PNG")

# --- File Uploader ---
uploaded_file = st.file_uploader("Drop your optical surface image here...", type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    # Read the image
    image = PIL.Image.open(uploaded_file)
    
    # Create two columns for Before/After comparison
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Original Surface")
        st.image(image, use_container_width=True)
        
    with col2:
        st.subheader("AI Defect Map")
        with st.spinner("Analyzing surface topography..."):
            # Run inference using the user-defined confidence threshold
            results = model.predict(image, conf=conf_threshold)
            
            # Extract the plotted image (BGR to RGB conversion handled by Streamlit/PIL)
            res_plotted = results[0].plot()
            st.image(res_plotted, use_container_width=True)

    # --- Analytics Dashboard ---
    st.divider()
    st.header("📊 Inspection Report")
    
    # Extract data from the YOLO results object
    defect_count = len(results[0].boxes)
    
    # Calculate average confidence if defects exist
    if defect_count > 0:
        avg_conf = np.mean(results[0].boxes.conf.cpu().numpy())
        status = "❌ Irregular (Defects Found)"
        status_color = "error"
    else:
        avg_conf = 0.0
        status = "✅ Regular (Clean Surface)"
        status_color = "success"
    
    # Display metrics
    m1, m2, m3 = st.columns(3)
    m1.metric("Surface Status", status)
    m2.metric("Total Defects Detected", defect_count)
    m3.metric("AI Confidence Level", f"{avg_conf:.1%}")

    # Display dynamic alerts
    if defect_count > 0:
        st.error(f"Attention: {defect_count} anomalies were detected on this surface. Quality control review is required.")
    else:
        st.success("Pass: The optical surface aligns with regularity standards. No defects mapped.")
