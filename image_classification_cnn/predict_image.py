import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# -----------------------------
# Settings
# -----------------------------
IMG_SIZE = (128, 128)
MODEL_PATH = "cnn_model.keras"

# -----------------------------
# Load model
# -----------------------------
model = tf.keras.models.load_model(MODEL_PATH)
print("✅ Model loaded")

# -----------------------------
# Prediction function
# -----------------------------
def predict_image(img_path):
    if not os.path.exists(img_path):
        print(f"❌ Image not found: {img_path}")
        return

    img = image.load_img(img_path, target_size=IMG_SIZE)
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)[0][0]

    if prediction >= 0.5:
        print(f"🟣 Prediction: THANOS ({prediction:.2f})")
    else:
        print(f"🟢 Prediction: JOKER ({1 - prediction:.2f})")


# -----------------------------
# TEST IMAGES (USE REAL PATHS)
# -----------------------------
predict_image("dataset/test/joker/j1.jpg")
predict_image("dataset/test/thanos/t1.jpg")
