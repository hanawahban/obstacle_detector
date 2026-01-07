import cv2 as cv
from src.camera import start_camera

cap = start_camera()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    cv.imshow("Camera Test", frame)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()
