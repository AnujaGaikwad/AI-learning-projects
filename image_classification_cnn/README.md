Joker vs Thanos – Image Classification using CNN

This project demonstrates **image classification using a Convolutional Neural Network (CNN)** to distinguish between two characters: **Joker** and **Thanos**.  
It is part of my **AI Learning Series**, focused on understanding deep learning concepts through hands-on projects.

---
📌 Project Overview

The goal of this project is to train a CNN model that can:
- Learn visual features from images
- Classify a given image as **Joker** or **Thanos**
- Predict the class of new, unseen images with confidence

This project covers:
- Image preprocessing
- Data augmentation
- CNN model building
- Model training & saving
- Image prediction using a trained model

---

🧠 Technologies Used

- **Python**
- **TensorFlow / Keras**
- **NumPy**
- **OpenCV (optional for future extensions)**

---

📂 Project Structure

image_classification_cnn/
│
├── dataset/
│ ├── train/
│ │ ├── joker/
│ │ └── thanos/
│ │
│ └── test/
│ ├── joker/
│ └── thanos/
│
├── train_cnn.py # Training the CNN model
├── predict_image.py # Predicting a single image
├── cnn_model.keras # Trained CNN model
└── README.md

