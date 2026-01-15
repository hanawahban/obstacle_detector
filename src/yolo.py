from ultralytics import YOLO

model = YOLO("yolov8n.pt")

# Expanded list of potential obstacles for visually impaired navigation
TARGET_CLASSES = [
    # People and animals
    "person", "dog", "cat", "bicycle", "motorcycle",
    
    # Furniture
    "chair", "couch", "bench", 
    
    # Large objects
    "car", "truck", "bus",
    
    # indoor obstacles
    "suitcase", "backpack", "handbag",
    
    # Outdoor obstacles
    "fire hydrant", "parking meter", "stop sign",
    
    # Other potential obstacles
    "potted plant", "umbrella"
]

# Objects that are typically elevated and less likely to be obstacles
ELEVATED_OBJECTS = ["table", "desk", "tv", "laptop", "keyboard"]

def detect_objects(frame):
    results = model(frame, stream=True)

    detections = []

    for result in results:
        for box in result.boxes:
            cls_id = int(box.cls[0])
            label = model.names[cls_id]
            conf = float(box.conf[0])

            # Check if object is in target classes or elevated objects
            if conf > 0.5:
                if label in TARGET_CLASSES:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    detections.append((label, conf, x1, y1, x2, y2))
                elif label in ELEVATED_OBJECTS:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    detections.append((f"{label} (elevated)", conf, x1, y1, x2, y2))

    return detections