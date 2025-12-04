import cv2

# Load Haar Cascade face detector
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# Open webcam (0 = default camera)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Cannot open camera")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Failed to grab frame")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=5
    )

    h, w, _ = frame.shape

    for (x, y, fw, fh) in faces:
        # Draw rectangle
        cv2.rectangle(frame, (x, y), (x+fw, y+fh), (0, 255, 0), 2)

        # Compute face center
        cx = x + fw // 2
        cy = y + fh // 2

        # Normalize features (0 to 1)
        cx_norm = cx / w
        cy_norm = cy / h
        size_norm = fw / w

        text = f"cx={cx_norm:.2f}, cy={cy_norm:.2f}, size={size_norm:.2f}"
        cv2.putText(frame, text, (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

    cv2.imshow("Face Detection Test", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
