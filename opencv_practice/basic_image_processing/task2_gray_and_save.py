import cv2
import os

# Ensure correct working directory
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Read image
img = cv2.imread("images/sample.png")

if img is None:
    raise FileNotFoundError("images/sample.png not found")

# Convert to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Save grayscale image
cv2.imwrite("images/gray_image.png", gray)

# Show images
cv2.imshow("Original", img)
cv2.imshow("Gray", gray)

cv2.waitKey(0)
cv2.destroyAllWindows()
