import streamlit as st
import cv2

st.set_page_config(layout="wide", page_title="Camera Feed")

# Title matching the requested text details style implies prominence
st.title("Camera View")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Model Information")
    # Display the specific details requested
    st.code("best_model=YOLO('runs/classify/train2/weights/best.pt')", language="python")

with col2:
    st.subheader("Live Feed")
    run = st.checkbox('Start Camera', value=True)
    frame_placeholder = st.empty()
    
    # Initialize camera
    camera = cv2.VideoCapture(0)
    
    while run:
        ret, frame = camera.read()
        if not ret:
            st.error("Could not capture video. Please ensure your webcam is connected.")
            break
            
        # Convert BGR to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Display the frame
        frame_placeholder.image(frame, channels="RGB")
        
    camera.release()
