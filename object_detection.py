import cv2
import streamlit as st
from PIL import Image
import numpy as np
from ultralytics import YOLO
import tempfile
import time

# Page configuration
st.set_page_config(
    page_title="Real-Time Object Detection",
    page_icon="🔍",
    layout="wide"
)

# Sidebar
st.sidebar.title("Settings")
model_type = st.sidebar.selectbox(
    "Select YOLO Model",
    ["YOLOv8n (Nano)", "YOLOv8s (Small)", "YOLOv8m (Medium)", "YOLOv8l (Large)", "YOLOv8x (Extra Large)"]
)

# Model selection mapping
model_mapping = {
    "YOLOv8n (Nano)": "yolov8n.pt",
    "YOLOv8s (Small)": "yolov8s.pt",
    "YOLOv8m (Medium)": "yolov8m.pt",
    "YOLOv8l (Large)": "yolov8l.pt",
    "YOLOv8x (Extra Large)": "yolov8x.pt"
}

# Confidence threshold
confidence_threshold = st.sidebar.slider(
    "Confidence Threshold", 0.0, 1.0, 0.25, 0.01
)

# Load the selected YOLO model
@st.cache_resource
def load_model(model_name):
    return YOLO(model_name)

model = load_model(model_mapping[model_type])

# Main app
st.title("Real-Time Object Detection with YOLOv8")
st.write("Detect objects in images, videos, or live webcam feed")

# Input selection
input_option = st.radio(
    "Select input type:",
    ["Image", "Video", "Webcam"]
)

# Function to draw bounding boxes
def draw_boxes(image, results):
    for result in results:
        for box in result.boxes:
            # Get box coordinates
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            
            # Get confidence and class
            conf = box.conf[0]
            cls = int(box.cls[0])
            label = f"{model.names[cls]} {conf:.2f}"
            
            # Draw rectangle and label
            color = (0, 255, 0)  # Green
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            cv2.putText(image, label, (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return image

# Image detection
if input_option == "Image":
    uploaded_file = st.file_uploader(
        "Upload an image", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        image = np.array(image)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.image(image, caption="Original Image", use_column_width=True)
        
        with col2:
            # Process image
            start_time = time.time()
            results = model.predict(image, conf=confidence_threshold)
            processed_image = draw_boxes(image.copy(), results)
            end_time = time.time()
            
            st.image(processed_image, caption="Processed Image", use_column_width=True)
            st.write(f"Detection time: {(end_time - start_time):.2f} seconds")
            
            # Display detected objects
            detected_objects = []
            for result in results:
                for box in result.boxes:
                    cls = int(box.cls[0])
                    detected_objects.append(model.names[cls])
            
            if detected_objects:
                st.write("Detected objects:", ", ".join(detected_objects))

# Video detection
elif input_option == "Video":
    uploaded_file = st.file_uploader(
        "Upload a video", type=["mp4", "avi", "mov"])
    
    if uploaded_file is not None:
        # Save uploaded video to temporary file
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())
        
        # Open video file
        cap = cv2.VideoCapture(tfile.name)
        
        stframe = st.empty()
        
        # Process video frame by frame
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Convert BGR to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process frame
            results = model.predict(frame, conf=confidence_threshold)
            processed_frame = draw_boxes(frame.copy(), results)
            
            # Display processed frame
            stframe.image(processed_frame, channels="RGB")
        
        cap.release()

# Webcam detection
elif input_option == "Webcam":
    run = st.checkbox("Start Webcam")
    FRAME_WINDOW = st.image([])
    cap = cv2.VideoCapture(0)
    
    while run:
        ret, frame = cap.read()
        if not ret:
            st.error("Failed to access webcam")
            break
        
        # Convert BGR to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process frame
        results = model.predict(frame, conf=confidence_threshold)
        processed_frame = draw_boxes(frame.copy(), results)
        
        # Display processed frame
        FRAME_WINDOW.image(processed_frame, channels="RGB")
    
    cap.release()

# Footer
st.markdown("---")
st.markdown(
    """
    **GitHub:** [Project Repository](https://github.com/yourusername/yolov8-streamlit-app)  
    **Connect with me on [LinkedIn](https://linkedin.com/in/yourprofile)**
    """
)
