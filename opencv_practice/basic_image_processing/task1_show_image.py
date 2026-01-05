import cv2
import os

# Force working directory to project root
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

img = cv2.imread("images/sample.png")

if img is None:
    raise FileNotFoundError("images/sample.png not found")

cv2.imshow("Original Image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
