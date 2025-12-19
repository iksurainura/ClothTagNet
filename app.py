import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import pandas as pd
import time

# Set page configuration
st.set_page_config(
    page_title="Real-time Object Detection",
    page_icon="🧵",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for UI enhancements
st.markdown("""
<style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .block-container { padding-top: 2rem; padding-bottom: 2rem; }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    """Load the YOLO model."""
    try:
        model = YOLO(r'runs/detect/train/weights/best.pt')
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

def main():
    # Header Section
    st.title("🧵 Real-time Cloth Tag Detection")
    
    col_text, col_toggle = st.columns([3, 1])
    with col_text:
        st.markdown("##### Intelligent Tag Identification & Analysis")
    with col_toggle:
        run_camera = st.toggle("Start Camera", value=False)

    model = load_model()
    if model is None: return

    st.divider()

    # Main Content Area
    col_video, col_stats = st.columns([1, 1])

    with col_video:
        st.caption("Live Feed")
        camera_placeholder = st.empty()

    with col_stats:
        st.caption("Detection Results")
        # Persistent placeholder for the colored status output
        status_container = st.empty()

    # Logic
    if run_camera:
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            st.error("Could not open video device.")
            return

        last_update_time = time.time()
        # Track state to update UI
        current_prediction = "none" # Default state

        while run_camera:
            ret, frame = cap.read()
            if not ret: break

            results = model(frame, conf=0.25, verbose=False)
            annotated_frame = frame.copy()
            
            # Reset detection state for the current frame
            frame_detection = "none"

            for r in results:
                for box in r.boxes:
                    cls = int(box.cls[0])
                    
                    # Core Logic: Mapping Class ID
                    if cls == 1:
                        frame_detection = "good"
                    elif cls == 0:
                        frame_detection = "damaged"

                    # Drawing logic on frame
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    color = (0, 255, 0) if cls == 1 else (0, 0, 255)
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 3)

            # Update the Color and Text every 2 seconds
            if time.time() - last_update_time > 2.0:
                if frame_detection == "good":
                    status_container.success("### ✅ STATUS: GOOD LABEL")
                elif frame_detection == "damaged":
                    status_container.error("### ❌ STATUS: DAMAGED LABEL")
                else:
                    status_container.info("### 🔍 STATUS: NO TAG DETECTED")
                
                last_update_time = time.time()

            # Display Video
            frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            camera_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)
            
        cap.release()
    else:
        with col_video:
            st.info("Camera is off. Toggle the switch above to start.")
        status_container.write("System Standby")

if __name__ == '__main__':
    main()