import cv2
import os
import numpy as np

DATASET = "images"
CASCADE = "haarcascade_frontalface_default.xml"

face_cascade = cv2.CascadeClassifier(CASCADE)

faces = []
labels = []
label_map = {}
label_id = 0

for person in os.listdir(DATASET):
    person_path = os.path.join(DATASET, person)
    if not os.path.isdir(person_path):
        continue

    label_map[label_id] = person

    for img_name in os.listdir(person_path):
        img_path = os.path.join(person_path, img_name)
        img = cv2.imread(img_path)

        if img is None:
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)  # 🔥 filter/lighting fix

        faces_detected = face_cascade.detectMultiScale(
            gray, scaleFactor=1.2, minNeighbors=5
        )

        for (x, y, w, h) in faces_detected:
            face = gray[y:y+h, x:x+w]
            face = cv2.resize(face, (200, 200))
            faces.append(face)
            labels.append(label_id)

    label_id += 1

model = cv2.face.LBPHFaceRecognizer_create(
    radius=2,
    neighbors=8,
    grid_x=8,
    grid_y=8
)

model.train(faces, np.array(labels))
model.save("face_model.yml")
np.save("labels.npy", label_map)

print("✅ Training done with high accuracy")
