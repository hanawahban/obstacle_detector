import cv2
from src.camera import start_camera
from src.yolo import detect_objects
from src.audio_alerts import speak

cap = start_camera()

# frame dimensions 
ret, test_frame = cap.read()
if not ret:
    raise RuntimeError("Cannot read from camera")

FRAME_HEIGHT = test_frame.shape[0]
FRAME_WIDTH = test_frame.shape[1]
FRAME_AREA = FRAME_HEIGHT * FRAME_WIDTH

# define the ground level objects
GROUND_LEVEL_THRESHOLD = FRAME_HEIGHT * 0.5

while True:
    ret, frame = cap.read()
    if not ret:
        break

    detections = detect_objects(frame)
    
    # Track obstacles that need alerting
    priority_obstacles = []

    for label, conf, x1, y1, x2, y2 in detections:
        # Calculate bounding box 
        box_area = (x2 - x1) * (y2 - y1)
        area_ratio = box_area / FRAME_AREA
        
        # Calculate center points
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        bottom_y = y2  # Bottom of bounding box
        
        # Check if object is at ground level 
        is_ground_level = bottom_y > GROUND_LEVEL_THRESHOLD
        
        # Calculate horizontal position
        relative_x = center_x / FRAME_WIDTH
        
        # Determine if object is in forward path (center 40% of frame)
        is_in_forward_path = 0.3 < relative_x < 0.7
        
        # Determine direction
        if center_x < FRAME_WIDTH / 3:
            direction = "on your left"
        elif center_x > 2 * FRAME_WIDTH / 3:
            direction = "on your right"
        else:
            direction = "ahead"
        
        # Determine proximity based on area ratio
        if area_ratio > 0.25:
            proximity = "very close"
            distance_priority = 3
        elif area_ratio > 0.12:
            proximity = "close"
            distance_priority = 2
        elif area_ratio > 0.05:
            proximity = "approaching"
            distance_priority = 1
        else:
            proximity = None
            distance_priority = 0
        
        # Draw bounding box 
        box_color = (0, 255, 0) if is_ground_level else (0, 255, 255)
        cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
        
        # Add label
        label_text = f"{label} {proximity if proximity else 'far'}"
        cv2.putText(frame, label_text, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, box_color, 2)
        
        if is_ground_level and proximity in ["close", "very close", "approaching"]:
            # Prioritize objects in forward path
            path_priority = 2 if is_in_forward_path else 1
            
            obstacle_info = {
                'label': label,
                'direction': direction,
                'proximity': proximity,
                'priority': distance_priority * path_priority,
                'area_ratio': area_ratio
            }
            priority_obstacles.append(obstacle_info)
    
    # Alert for the highest priority obstacle only
    if priority_obstacles:
        # Sort by priority (highest first)
        priority_obstacles.sort(key=lambda x: x['priority'], reverse=True)
        top_obstacle = priority_obstacles[0]
        
        #alert message
        if top_obstacle['proximity'] == "very close":
            alert = f"Warning! {top_obstacle['label']} very close {top_obstacle['direction']}"
        else:
            alert = f"{top_obstacle['label']} {top_obstacle['direction']}"
        
        speak(alert)
    
    # Visual feedback
    cv2.imshow("Obstacle Detector", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()