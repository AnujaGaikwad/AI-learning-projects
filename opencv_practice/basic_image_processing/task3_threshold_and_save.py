import cv2
import os

# Absolute path of the current script (basic_image_processing)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Correct path to image
image_path = os.path.join(BASE_DIR, "images", "sample.png")

# Read image
img = cv2.imread(image_path)


# Convert to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Apply thresholding
_, thresh = cv2.threshold(gray, 199, 255, cv2.THRESH_BINARY)

# Save threshold image
cv2.imwrite(os.path.join(BASE_DIR, "images", "threshold_image.png"), thresh)

# Show results
cv2.imshow("Original", img)
cv2.imshow("Gray", gray)
cv2.imshow("Threshold", thresh)

cv2.waitKey(0)
cv2.destroyAllWindows()
