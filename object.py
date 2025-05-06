import cv2
import matplotlib.pyplot as plt

# Load pre-trained model and config file
net = cv2.dnn.readNetFromCaffe('C:/prj2/deploy.prototxt', 'C:/prj2/mobilenet_iter_73000.caffemodel')



# List of object classes MobileNet SSD was trained on
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
           "sofa", "train", "tvmonitor"]

# Start the webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Prepare the frame for object detection
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)),
                                 0.007843, (300, 300), 127.5)
    
    net.setInput(blob)
    detections = net.forward()

    # Loop over the detections
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        if confidence > 0.5:
            idx = int(detections[0, 0, i, 1])
            label = CLASSES[idx]
            
            box = detections[0, 0, i, 3:7] * [w, h, w, h]
            (startX, startY, endX, endY) = box.astype("int")

            # Draw bounding box and label
            cv2.rectangle(frame, (startX, startY), (endX, endY),
                          (0, 255, 0), 2)
            cv2.putText(frame, f"{label}: {confidence:.2f}",
                        (startX, startY - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display the output
    cv2.imshow("Object Detection", frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # Inside the loop, replace cv2.imshow with plt.imshow
plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
plt.title("Object Detection")
plt.show()

cap.release()
cv2.destroyAllWindows()
