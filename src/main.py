import cv2
from src.camera import start_camera
from src.yolo import detect_objects
from src.audio_alerts import speak

cap = start_camera()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    detections = detect_objects(frame)
    frame_area = frame.shape[0] * frame.shape[1]

    for label, conf, x1, y1, x2, y2 in detections:
        # bounding box
        box_area = (x2 - x1) * (y2 - y1)
        area_ratio = box_area / frame_area

        # proximity
        if area_ratio > 0.25:
            proximity = "very close"
        elif area_ratio > 0.12:
            proximity = "close"
        else:
            proximity = "far"

        frame_width = frame.shape[1]
        center_x = (x1 + x2) / 2

        if center_x < frame_width / 3:
            direction = "left"
        elif center_x > 2 * frame_width / 3:
            direction = "right"
        else:
            direction = "ahead"

        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Audio alert
        if proximity in ["close", "very close"]:
            speak(f"{label} {direction}")

    cv2.imshow("Obstacle Detector", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
