**Obstacle Detection System**

An AI-powered real-time obstacle detection system designed for assistive applications.
The system uses YOLO object detection, OpenCV, and audio alerts to identify obstacles in front of the user and provide real-time feedback.

This project is optimized for NVIDIA Jetson Orin Nano.

**Features**
- Real-time camera input (USB / IP / phone camera)
  
- Object detection using YOLO (Ultralytics)
  
- Detection of common obstacles (people, furniture, objects)
  
- Audio alerts for nearby obstacles

- Camera-only proximity estimation (no external sensors)

- Optimized for Jetson Orin Nano (GPU acceleration)

**Tech Stack**
- Python 3
  
- OpenCV
  
- YOLOv8 (Ultralytics)
  
- PyTorch
  
- Text-to-Speech (audio alerts)
  
- NVIDIA JetPack

**Installation**
1. Clone the repository
   
3. Create and activate virtual environment
   
5. Install dependencies
   
7. Run the program

**How it works**

1. Camera captures real-time video frames
   
2. YOLO detects objects in each frame
   
3. Bounding box size is used to estimate proximity

4. Audio alerts trigger when obstacles are close

5. Visual feedback is displayed on screen
