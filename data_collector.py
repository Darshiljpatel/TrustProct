import cv2
import csv
import os
from datetime import datetime

# Load Face Detector
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# Create output folder + CSV
os.makedirs("data", exist_ok=True)
csv_path = "data/features.csv"

file_exists = os.path.isfile(csv_path)

csv_file = open(csv_path, "a", newline="")
writer = csv.writer(csv_file)

if not file_exists:
    writer.writerow(["cx_norm", "cy_norm", "size_norm", "face_visible", "label", "timestamp"])

# Open webcam
cap = cv2.VideoCapture(0)

print("✅ Press:")
print("   O = ON SCREEN")
print("   F = OFF SCREEN")
print("   Q = Quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    h, w, _ = frame.shape
    label = None

    key = cv2.waitKey(1) & 0xFF

    if key == ord("o"):
        label = 0   # ON_SCREEN
    elif key == ord("f"):
        label = 1   # OFF_SCREEN
    elif key == ord("q"):
        break

    if len(faces) > 0:
        x, y, fw, fh = faces[0]

        cx = x + fw // 2
        cy = y + fh // 2

        cx_norm = cx / w
        cy_norm = cy / h
        size_norm = fw / w
        face_visible = 1

        cv2.rectangle(frame, (x, y), (x+fw, y+fh), (0, 255, 0), 2)

    else:
        cx_norm = -1
        cy_norm = -1
        size_norm = 0
        face_visible = 0

    if label is not None:
        writer.writerow([
            cx_norm, cy_norm, size_norm, face_visible,
            label, datetime.now().isoformat()
        ])
        print("✅ Data Saved → Label:", label)

    cv2.putText(frame, "O = ON | F = OFF | Q = QUIT",
                (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (0, 0, 255), 2)

    cv2.imshow("DATA COLLECTION", frame)

cap.release()
csv_file.close()
cv2.destroyAllWindows()
