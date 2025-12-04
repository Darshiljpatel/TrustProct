import cv2
import numpy as np
from joblib import load
from collections import deque

# Load model + scaler
model = load("model/cheat_model.joblib")
scaler = load("model/scaler.joblib")

print("✅ Model loaded successfully")

# Load face detector
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# Maintain last 30 predictions for smooth risk score
risk_window = deque(maxlen=30)

# Open webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Cannot open webcam")
    exit()

def get_risk_label(prob):
    if prob < 0.3:
        return "LOW", (0, 255, 0)     # Green
    elif prob < 0.6:
        return "MEDIUM", (0, 255, 255)  # Yellow
    else:
        return "HIGH", (0, 0, 255)    # Red

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    # Default values if no face found
    cx_norm, cy_norm, size_norm, face_visible = -1, -1, 0, 0

    if len(faces) > 0:
        (x, y, fw, fh) = faces[0]  # First face only

        cx = x + fw // 2
        cy = y + fh // 2

        cx_norm = cx / w
        cy_norm = cy / h
        size_norm = fw / w
        face_visible = 1

        cv2.rectangle(frame, (x, y), (x + fw, y + fh), (0, 255, 0), 2)
    else:
        face_visible = 0

    # Prepare input for model
    X = np.array([[cx_norm, cy_norm, size_norm, face_visible]])
    X_scaled = scaler.transform(X)

    # Model prediction probability of OFF SCREEN (cheating)
    prob_off = model.predict_proba(X_scaled)[0][1]

    # Store probability
    risk_window.append(prob_off)

    # Smooth risk using last 30 frames
    avg_risk = np.mean(risk_window)

    # Get label + color
    risk_label, color = get_risk_label(avg_risk)

    # Display risk
    cv2.putText(frame, f"Risk: {avg_risk:.2f} ({risk_label})",
                (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)

    # Face visible indicator
    cv2.putText(frame, f"Face Visible: {face_visible}",
                (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

    cv2.imshow("Cheat Detection System", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
