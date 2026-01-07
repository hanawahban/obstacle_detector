import cv2
from src.camera import start_camera
from src.yolo import detect_objects

cap = start_camera()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    detections = detect_objects(frame)

    for label, conf, x1, y1, x2, y2 in detections:
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
        cv2.putText(frame, label, (x1, y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

    cv2.imshow("YOLO Test", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
