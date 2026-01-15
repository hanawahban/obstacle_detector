
import cv2
import numpy as np
import time
from src.camera import start_camera
from src.yolo import detect_objects
from src.audio_alerts import speak
from src.depth import DepthEstimator

# Performance settings
DISPLAY_ENABLED = True  
DEPTH_FRAME_SKIP = 3  
TARGET_FPS = 30
DETECTION_CONFIDENCE = 0.5

cap = start_camera()
depth_estimator = DepthEstimator(model_type="MiDAS_small")

ret, test_frame = cap.read()
if not ret:
    raise RuntimeError("Cannot read from camera")

# Resize input for faster processing
PROCESS_WIDTH = 640
PROCESS_HEIGHT = 480
FRAME_HEIGHT = PROCESS_HEIGHT
FRAME_WIDTH = PROCESS_WIDTH

GROUND_LEVEL_THRESHOLD = FRAME_HEIGHT * 0.6

frame_count = 0
current_depth_map = None
fps_counter = []

print("Obstacle detection system ready!")
print(f"Processing at {PROCESS_WIDTH}x{PROCESS_HEIGHT}")
print(f"Display: {'Enabled' if DISPLAY_ENABLED else 'Disabled (headless)'}")

try:
    while True:
        loop_start = time.time()
        
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = cv2.resize(frame, (PROCESS_WIDTH, PROCESS_HEIGHT))
        
        frame_count += 1
        
        # Update depth map 
        if frame_count % DEPTH_FRAME_SKIP == 0:
            depth_start = time.time()
            current_depth_map, current_depth_normalized = depth_estimator.estimate_depth(frame)
            depth_time = time.time() - depth_start
        
        # Detect objects
        detect_start = time.time()
        detections = detect_objects(frame)
        detect_time = time.time() - detect_start
        
        priority_obstacles = []

        for label, conf, x1, y1, x2, y2 in detections:
            if conf < DETECTION_CONFIDENCE:
                continue
            
            center_x = (x1 + x2) / 2
            bottom_y = y2
            
            is_ground_level = bottom_y > GROUND_LEVEL_THRESHOLD
            
            # Get depth info
            if current_depth_map is not None and is_ground_level:
                obj_depth, depth_category = depth_estimator.get_object_depth(
                    current_depth_map, x1, y1, x2, y2
                )
            else:
                continue  
            
            # Calculate position
            relative_x = center_x / FRAME_WIDTH
            is_in_forward_path = 0.3 < relative_x < 0.7
            
            # Determine direction
            if center_x < FRAME_WIDTH / 3:
                direction = "on your left"
            elif center_x > 2 * FRAME_WIDTH / 3:
                direction = "on your right"
            else:
                direction = "ahead"
            
            if depth_category == "very_close":
                proximity = "very close"
                distance_priority = 4
            elif depth_category == "close":
                proximity = "close"
                distance_priority = 3
            elif depth_category == "medium":
                proximity = "approaching"
                distance_priority = 2
            else:
                continue  # Ignore far objects
            
            path_priority = 2 if is_in_forward_path else 1
            
            # Only alert on significant threats
            if depth_category in ["very_close", "close", "medium"]:
                obstacle_info = {
                    'label': label,
                    'direction': direction,
                    'proximity': proximity,
                    'priority': distance_priority * path_priority,
                    'in_path': is_in_forward_path
                }
                priority_obstacles.append(obstacle_info)
            
            # Draw on frame if display enabled
            if DISPLAY_ENABLED:
                if depth_category == "very_close":
                    color = (0, 0, 255)  
                elif depth_category == "close":
                    color = (0, 165, 255)  
                else:
                    color = (0, 255, 255)  
                
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f"{label}: {proximity}", (x1, y1-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        # Alert for highest priority obstacle only
        if priority_obstacles:
            priority_obstacles.sort(key=lambda x: x['priority'], reverse=True)
            top = priority_obstacles[0]
            
            if top['proximity'] == "very close":
                alert = f"Warning! {top['label']} very close {top['direction']}"
            else:
                alert = f"{top['label']} {top['direction']}"
            
            speak(alert)
        
        if DISPLAY_ENABLED:
            loop_time = time.time() - loop_start
            current_fps = 1.0 / loop_time if loop_time > 0 else 0
            fps_counter.append(current_fps)
            if len(fps_counter) > 30:
                fps_counter.pop(0)
            avg_fps = np.mean(fps_counter)
            
            cv2.putText(frame, f"FPS: {avg_fps:.1f}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            if current_depth_normalized is not None:
                depth_colored = depth_estimator.visualize_depth(current_depth_normalized)
                combined = np.hstack((frame, depth_colored))
                cv2.imshow("Obstacle Detector", combined)
            else:
                cv2.imshow("Obstacle Detector", frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            loop_time = time.time() - loop_start
            target_time = 1.0 / TARGET_FPS
            if loop_time < target_time:
                time.sleep(target_time - loop_time)

except KeyboardInterrupt:
    print("\nShutting down...")

finally:
    cap.release()
    if DISPLAY_ENABLED:
        cv2.destroyAllWindows()
    print("Obstacle detection system stopped.")