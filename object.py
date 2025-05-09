import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import cv2
import numpy as np

# Title of the app
st.title("🎥 Live Object Detection with Webcam")

# Load the model only once to avoid reloading on every frame
try:
    net = cv2.dnn.readNetFromCaffe('deploy.prototxt', 'mobilenet_iter_73000.caffemodel')
except Exception as e:
    st.error(f"Error loading model files: {e}")
    st.stop()

# List of classes MobileNet SSD was trained on
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
           "sofa", "train", "tvmonitor"]

# Define the object detection class
class ObjectDetector(VideoTransformerBase):
    def transform(self, frame):
        try:
            img = frame.to_ndarray(format="bgr24")
            h, w = img.shape[:2]

            blob = cv2.dnn.blobFromImage(cv2.resize(img, (300, 300)),
                                         0.007843, (300, 300), 127.5)

            net.setInput(blob)
            detections = net.forward()

            for i in range(detections.shape[2]):
                confidence = detections[0, 0, i, 2]
                if confidence > 0.5:
                    idx = int(detections[0, 0, i, 1])
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    (startX, startY, endX, endY) = box.astype("int")

                    label = CLASSES[idx] if idx < len(CLASSES) else "unknown"
                    cv2.rectangle(img, (startX, startY), (endX, endY), (0, 255, 0), 2)
                    cv2.putText(img, f"{label}: {confidence:.2f}", (startX, startY - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            return img
        except Exception as e:
            print(f"Frame processing error: {e}")
            return frame.to_ndarray(format="bgr24")  # fallback to unprocessed frame

# Launch webcam and apply object detection
try:
    webrtc_streamer(
        key="live",
        video_transformer_factory=ObjectDetector,
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True
    )
except Exception as e:
    st.error(f"An error occurred while initializing the camera: {e}")
