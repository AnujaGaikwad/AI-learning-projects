import cv2
import os

# Absolute path to this script's directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Absolute path to image
IMAGE_PATH = os.path.join(BASE_DIR, "images", "sample.png")

# Read image
img = cv2.imread(IMAGE_PATH)

if img is None:
    raise FileNotFoundError(f"Image not found at: {IMAGE_PATH}")

# Show image
cv2.imshow("Original Image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
