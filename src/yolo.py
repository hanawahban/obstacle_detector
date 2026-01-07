from ultralytics import YOLO

model = YOLO("yolov5n.pt")

TARGET_CLASSES = ["person", "chair", "table"]

def detect_objects(frame):
    results = model(frame, stream=True)

    detections = []

    for result in results:
        for box in result.boxes:
            cls_id = int(box.cls[0])
            label = model.names[cls_id]
            conf = float(box.conf[0])

            if label in TARGET_CLASSES and conf > 0.5:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                detections.append((label, conf, x1, y1, x2, y2))

    return detections
