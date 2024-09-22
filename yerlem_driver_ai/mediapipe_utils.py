import mediapipe as mp
import cv2

class MediaPipeUtils:
    def __init__(self):
        # DESCRIPTION: Initialize MediaPipe modules.
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.mp_face_mesh = mp.solutions.face_mesh
        # self.mp_hands = mp.solutions.hands
        
        # DESCRIPTION: Initialize face mesh and hands modules.
        self.face_mesh = self.mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)
        # self.hands = self.mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)

    # DESCRIPTION: Process the image to detect face and hand landmarks.
    def process(self, image):
        # DESCRIPTION: Convert the image to RGB format. Because MediaPipe processes RGB images.
        image.flags.writeable = False
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        face_results = self.face_process(rgb_image)
        # hand_results = self.hand_process(rgb_image)

        # DESCRIPTION: Convert the image back to BGR format. Because OpenCV processes BGR images.
        image.flags.writeable = True
        image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)

        return face_results # , hand_results

    # DESCRIPTION: Process the image to detect face landmarks.
    def face_process(self, rgb_image):
        face_results = self.face_mesh.process(rgb_image)
        return face_results
    
    # DESCRIPTION: Process the image to detect hand landmarks.
    # def hand_process(self, rgb_image):
    #     hand_results = self.hands.process(rgb_image)
    #     return hand_results