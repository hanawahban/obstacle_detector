import cv2
import numpy as np
from src.camera import start_camera
from src.depth import DepthEstimator

print("Initializing depth estimation test...")
cap = start_camera()
depth_estimator = DepthEstimator(model_type="MiDAS_small")

print("\nTest Controls:")
print("  Q - Quit")
print("  S - Save current frame and depth map")
print("  SPACE - Pause/Resume")
print("  Click on image to see depth value at that point")

paused = False
click_x, click_y = -1, -1
current_depth_map = None

def mouse_callback(event, x, y, flags, param):
    global click_x, click_y
    if event == cv2.EVENT_LBUTTONDOWN:
        click_x, click_y = x, y

cv2.namedWindow("Depth Test")
cv2.setMouseCallback("Depth Test", mouse_callback)

frame_count = 0

try:
    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Get depth map
            current_depth_map, depth_normalized = depth_estimator.estimate_depth(frame)
            
            # Create colored depth visualization
            depth_colored = depth_estimator.visualize_depth(depth_normalized)
            
            # Resize for display
            frame_display = cv2.resize(frame, (640, 480))
            depth_display = cv2.resize(depth_colored, (640, 480))
            
            # If user clicked, show depth value
            if click_x >= 0 and click_y >= 0:
                # Map click coordinates to original depth map
                h, w = current_depth_map.shape
                map_x = int(click_x * w / 640)
                map_y = int(click_y * h / 480)
                
                if 0 <= map_x < w and 0 <= map_y < h:
                    depth_value = current_depth_map[map_y, map_x]
                    
                    # Draw crosshair on both views
                    cv2.drawMarker(frame_display, (click_x, click_y), 
                                 (0, 255, 0), cv2.MARKER_CROSS, 20, 2)
                    cv2.drawMarker(depth_display, (click_x, click_y), 
                                 (0, 255, 0), cv2.MARKER_CROSS, 20, 2)
                    
                    # Show depth value
                    depth_text = f"Depth: {depth_value:.2f}"
                    cv2.putText(frame_display, depth_text, (click_x + 10, click_y - 10),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Add statistics
            if current_depth_map is not None:
                depth_min = np.min(current_depth_map)
                depth_max = np.max(current_depth_map)
                depth_mean = np.mean(current_depth_map)
                
                stats_text = [
                    f"Frame: {frame_count}",
                    f"Min: {depth_min:.2f}",
                    f"Max: {depth_max:.2f}",
                    f"Mean: {depth_mean:.2f}"
                ]
                
                y_offset = 30
                for text in stats_text:
                    cv2.putText(frame_display, text, (10, y_offset),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    y_offset += 25
            
            # Add instruction text
            cv2.putText(depth_display, "Click to measure depth", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            
            # Combine views
            combined = np.hstack((frame_display, depth_display))
            cv2.imshow("Depth Test", combined)
        
        # Handle keyboard input
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            break
        elif key == ord('s'):
            # Save current frame and depth map
            cv2.imwrite(f"frame_{frame_count}.jpg", frame)
            cv2.imwrite(f"depth_{frame_count}.jpg", depth_colored)
            cv2.imwrite(f"depth_raw_{frame_count}.png", (depth_normalized).astype(np.uint8))
            print(f"Saved frame_{frame_count}.jpg and depth maps")
        elif key == ord(' '):
            paused = not paused
            print("Paused" if paused else "Resumed")
        elif key == ord('c'):
            # Clear click point
            click_x, click_y = -1, -1

except KeyboardInterrupt:
    print("\nTest interrupted by user")

finally:
    cap.release()
    cv2.destroyAllWindows()
    print("Depth test completed")