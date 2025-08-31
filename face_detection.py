# Robust Real-Time Face Detection with Haar Cascades and Confidence Filtering
import cv2
import os
from datetime import datetime

cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(cascade_path)
if face_cascade.empty():
    raise IOError("Error: Haar Cascade file not found!")

def detect_faces(frame, save_output=False, output_file="output.jpg"):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=6, minSize=(30, 30)
    )

    if len(faces) == 0:
        print("⚠ No faces detected.")
    else:
        print(f"✅ Detected {len(faces)} face(s).")

    for (x, y, w, h) in faces:
        if w * h < 4000:       color = (255, 0, 0)   # Small
        elif w * h < 10000:    color = (0, 255, 0)   # Medium
        else:                  color = (0, 0, 255)   # Large
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

    cv2.putText(frame, f"Faces: {len(faces)}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    if save_output:
        cv2.imwrite(output_file, frame)
        print(f"💾 Saved output image as {output_file}")

    return frame

img_path = "D:/Digital Pathshala/AI ML With Python/Projects/Robust Real-Time Face Detection/img"
static_image_path = os.path.join(img_path, "ram.jpg")
static_output_path = os.path.join(img_path, "ram_detected.jpg")

# Static Image Detection 
if os.path.exists(static_image_path):
    img = cv2.imread(static_image_path)
    if img is not None:
        result = detect_faces(img, save_output=True, output_file=static_output_path)
        cv2.imshow("Face Detection - Static Image", result)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("⚠ Could not read static image.")
else:
    print(f"⚠ Static image not found at {static_image_path}")


# Webcam Detection
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise IOError("Error: Cannot access webcam.")

print("🎥 Press 'q' to quit webcam mode, 's' to save current frame")
while True:
    ret, frame = cap.read()
    if not ret:
        print("⚠ Failed to capture frame.")
        break

    result = detect_faces(frame)
    cv2.imshow("Face Detection - Webcam", result)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    elif key == ord("s"):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        webcam_output_path = os.path.join(img_path, f"webcam_detected_{timestamp}.jpg")
        cv2.imwrite(webcam_output_path, result)
        print(f"💾 Saved current webcam frame as {webcam_output_path}")

cap.release()
cv2.destroyAllWindows()
