import cv2
import imutils

cam = cv2.VideoCapture(0)

while True:
    ret, frame = cam.read()
    if not ret:
        break

    frame = imutils.resize(frame, width=500)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (21, 21), 0)

    cv2.imshow("Gray", gray)
    cv2.imshow("Blurred", blur)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()
