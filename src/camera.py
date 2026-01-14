import cv2 as cv

def start_camera():
    cam = cv.VideoCapture(1)

    if not cam.isOpened():
        raise RuntimeError("Camera not accessible")

    return cam
