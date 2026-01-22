import cv2
import numpy as np
import pandas as pd
import os
from datetime import datetime

CASCADE = "haarcascade_frontalface_default.xml"
MODEL = "face_model.yml"
LABELS = "labels.npy"

face_cascade = cv2.CascadeClassifier(CASCADE)

model = cv2.face.LBPHFaceRecognizer_create()
model.read(MODEL)

label_map = np.load(LABELS, allow_pickle=True).item()

attendance = {}

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)  # 🔥 filter support

    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.2, minNeighbors=5
    )

    for (x, y, w, h) in faces:
        face = gray[y:y+h, x:x+w]
        face = cv2.resize(face, (200, 200))

        label, confidence = model.predict(face)

        if confidence < 95:   # 🔥 tuned threshold
            name = label_map[label]
            color = (0, 255, 0)

            if name not in attendance:
                attendance[name] = datetime.now().strftime("%H:%M:%S")

            text = f"{name} ({int(confidence)})"
        else:
            text = "Unknown"
            color = (0, 0, 255)

        cv2.putText(
            frame, text, (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2
        )

        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

    cv2.imshow("Face Attendance System", frame)

    if cv2.waitKey(1) == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()

# 🔥 Save attendance with DATE + TIME (append)
today = datetime.now().strftime("%Y-%m-%d")

rows = [[name, today, time] for name, time in attendance.items()]
df = pd.DataFrame(rows, columns=["Name", "Date", "Time"])

if os.path.exists("attendance.csv"):
    df.to_csv("attendance.csv", mode="a", header=False, index=False)
else:
    df.to_csv("attendance.csv", index=False)

print("✅ Attendance saved with date & time")
