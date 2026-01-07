import cv2 as cv

def start_camera():
    webcam = cv.VideoCapture(0)

    if not webcam.isOpened():
        raise RuntimeError("Camera not accessible")

    return webcam
