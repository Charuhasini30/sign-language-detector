from ultralytics import YOLO
import cv2
import numpy as np
from tensorflow import keras

# Load YOLO hand detection model (pretrained or custom)
yolo = YOLO('yolov8n.pt')  # or 'yolov8n-pose.pt'

# Load trained CNN model
model = keras.models.load_model("sign_lang_model.h5")

# Open webcam
cap = cv2.VideoCapture(0)

labels = [chr(i) for i in range(65, 91)]  # A–Z (if dataset is A–Y adjust accordingly)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # YOLO detection
    results = yolo(frame)
    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy()
        for box in boxes:
            x1, y1, x2, y2 = map(int, box)
            hand = frame[y1:y2, x1:x2]
            hand = cv2.cvtColor(hand, cv2.COLOR_BGR2GRAY)
            hand = cv2.resize(hand, (28, 28))
            hand = hand / 255.0
            hand = hand.reshape(1, 28, 28, 1)

            # Predict using trained CNN
            pred = np.argmax(model.predict(hand))
            label = labels[pred]

            # Draw results
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Sign Language Detection", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
