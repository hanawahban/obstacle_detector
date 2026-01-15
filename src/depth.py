import torch
import cv2
import numpy as np

class DepthEstimator:
    def __init__(self, model_type="MiDAS_small"):
        # Load model
        self.midas = torch.hub.load("intel-isl/MiDaS", model_type)
        
        # Set device (CUDA for Jetson)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.midas.to(self.device)
        self.midas.eval()
        
        # Load transforms
        midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
        
        if model_type == "DPT_Large" or model_type == "DPT_Hybrid":
            self.transform = midas_transforms.dpt_transform
        else:
            self.transform = midas_transforms.small_transform
        
        print(f"MiDaS model loaded on {self.device}")
    
    def estimate_depth(self, frame):
        # Convert BGR to RGB
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Apply transforms
        input_batch = self.transform(img_rgb).to(self.device)
        
        # Predict depth
        with torch.no_grad():
            prediction = self.midas(input_batch)
            
            # Resize to original resolution
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=img_rgb.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()
        
        depth_map = prediction.cpu().numpy()
        
        # Normalize for visualization (invert so closer = brighter)
        depth_map_normalized = cv2.normalize(depth_map, None, 0, 255, 
                                            norm_type=cv2.NORM_MINMAX, 
                                            dtype=cv2.CV_8U)
        
        # Invert: closer objects should have higher values
        depth_map_normalized = 255 - depth_map_normalized
        
        return depth_map, depth_map_normalized
    
    def get_object_depth(self, depth_map, x1, y1, x2, y2):
        # Extract region of interest
        roi = depth_map[y1:y2, x1:x2]
        
        # Calculate median depth (more robust than mean)
        median_depth = np.median(roi)
        
        # Get the closest point in the ROI (minimum depth value)
        min_depth = np.min(roi)
                
        if min_depth < np.percentile(depth_map, 10):  # Bottom 10% of depth values
            depth_category = "very_close"
        elif min_depth < np.percentile(depth_map, 25):  # Bottom 25%
            depth_category = "close"
        elif min_depth < np.percentile(depth_map, 50):  # Bottom 50%
            depth_category = "medium"
        else:
            depth_category = "far"
        
        return min_depth, depth_category
    
    def visualize_depth(self, depth_map_normalized):
        colored_depth = cv2.applyColorMap(depth_map_normalized, cv2.COLORMAP_PLASMA)
        
        return colored_depth