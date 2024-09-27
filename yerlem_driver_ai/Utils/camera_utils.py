import cv2

class CameraUtils:
    def __init__(self, camera_index=0):
        # DESCRIPTION: Initialize camera. `camera_index=0` is the default (first) camera.
        self.cap = cv2.VideoCapture(camera_index)

    # DESCRIPTION: Read a frame from the camera.
    def read_frame(self):
        # DESCRIPTION: `ret` is a boolean value that indicates whether the frame was read correctly.
        ret, image = self.cap.read()
        if not ret:
            print("Ignoring empty camera frame.")
        return cv2.flip(image, 1)

    # DESCRIPTION: Release the camera (close the camera).
    def release_camera(self):
        self.cap.release()