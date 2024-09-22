import cv2
from yerlem_driver_ai.mediapipe_utils import MediaPipeUtils
from yerlem_driver_ai.camera_utils import CameraUtils
from yerlem_driver_ai.json_utils import JsonUtils
from yerlem_driver_ai.drawing_utils import DrawingUtils
from yerlem_driver_ai.coordinate_finder import CoordinateFinder
from yerlem_driver_ai.coordinate_math_ops import CoordinateMathOps
from yerlem_driver_ai.models import Models
from yerlem_driver_ai.data_preprocessing1 import DataPreprocessing1
# from yerlem_driver_ai.rt_coordinates import RTCoordinates

class Application:
    def __init__(self):
        # DESCRIPTION: Initialize MediaPipeUtils, CameraUtils, JsonUtils, DrawingUtils, CoordinateFinder, and CoordinateMathOps classes.
        self.mp_utils = MediaPipeUtils()
        self.camera_utils = CameraUtils()
        self.json_utils = JsonUtils()
        self.drawing_utils = DrawingUtils(self.mp_utils.mp_drawing, self.mp_utils.mp_drawing_styles, self.mp_utils.mp_face_mesh) #, self.mp_utils.mp_hands
        self.coordinate_finder = CoordinateFinder()
        self.data_preprocessing1 = DataPreprocessing1()
        self.coordinate_math_ops = CoordinateMathOps(None,None,None,None,None,None)
        # self.rt_coordinates = RTCoordinates(None,None,None,None,None,None)
        self.models = Models()
        self.model_RF = self.models.import_model("random_forest_model_for_yawning_detection")

        # DESCRIPTION: Load face landmarks data and get the next ID from the JSON file.
        self.tfl_data = self.json_utils.load_data("data/raw/not_yawning.json")
        self.tfl_counter = self.json_utils.get_next_id(self.tfl_data)

    def run(self):
        while True:
            # DESCRIPTION: Read a frame from the camera and process it to detect face landmarks.
            image = self.camera_utils.read_frame()
            face_results = self.mp_utils.process(image)

            # DESCRIPTION: Estimating yawning.
            lips_points_coordinates = self.coordinate_finder.capture_points_coordinates(face_results, "lips_points", self.coordinate_finder.face_points["lips_points"])
            left_eye_points_coordinates = self.coordinate_finder.capture_points_coordinates(face_results, "left_eye_points", self.coordinate_finder.face_points["left_eye_points"])
            left_eyebrow_points_coordinates = self.coordinate_finder.capture_points_coordinates(face_results, "left_eyebrow_points", self.coordinate_finder.face_points["left_eyebrow_points"])
            right_eye_points_coordinates = self.coordinate_finder.capture_points_coordinates(face_results, "right_eye_points", self.coordinate_finder.face_points["right_eye_points"])
            right_eyebrow_points_coordinates = self.coordinate_finder.capture_points_coordinates(face_results, "right_eyebrow_points", self.coordinate_finder.face_points["right_eyebrow_points"])
            origin_points_coordinates = self.coordinate_finder.capture_points_coordinates(face_results, "origin_points", self.coordinate_finder.face_points["origin_points"])
            
            self.coordinate_math_ops.lips_points_coordinates = lips_points_coordinates
            self.coordinate_math_ops.left_eye_points_coordinates = left_eye_points_coordinates
            self.coordinate_math_ops.left_eyebrow_points_coordinates = left_eyebrow_points_coordinates
            self.coordinate_math_ops.right_eye_points_coordinates = right_eye_points_coordinates
            self.coordinate_math_ops.right_eyebrow_points_coordinates = right_eyebrow_points_coordinates
            self.coordinate_math_ops.origin_points_coordinates = origin_points_coordinates
            
            eyes_openness = self.coordinate_math_ops.calculate_eye_openness()
            lips_openness = self.coordinate_math_ops.calculate_lips_openness()
            eyebrow_eye_distance = self.coordinate_math_ops.calculate_eyebrow_eye_distance()
            eyebrow_slope = self.coordinate_math_ops.calculate_eyebrow_slope()
            lips_slope = self.coordinate_math_ops.calculate_lips_slope()

            captured_data = [{"id": 1,
                              "label": "Face Landmarks",
                              "landmarks": [eyes_openness + \
                                            lips_openness + \
                                            eyebrow_eye_distance + \
                                            eyebrow_slope + \
                                            lips_slope + \
                                            lips_points_coordinates + \
                                            left_eye_points_coordinates + \
                                            left_eyebrow_points_coordinates + \
                                            right_eye_points_coordinates + \
                                            right_eyebrow_points_coordinates]}]

            ready_data = self.data_preprocessing1.json_to_csv(self.data_preprocessing1.relabel(captured_data))
            yawning = self.model_RF.predict(ready_data)

            # DESCRIPTION: Draw face landmarks on the image.
            self.drawing_utils.draw_face_landmarks(image, face_results) #, self.coordinate_finder.face_points)

            # DESCRIPTION: If the 'c' key is pressed...
            if cv2.waitKey(1) & 0xFF == ord('c'):
                # DESCRIPTION: Capture face landmarks.
                lips_points_coordinates = self.coordinate_finder.capture_points_coordinates(face_results, "lips_points", self.coordinate_finder.face_points["lips_points"])
                left_eye_points_coordinates = self.coordinate_finder.capture_points_coordinates(face_results, "left_eye_points", self.coordinate_finder.face_points["left_eye_points"])
                left_eyebrow_points_coordinates = self.coordinate_finder.capture_points_coordinates(face_results, "left_eyebrow_points", self.coordinate_finder.face_points["left_eyebrow_points"])
                right_eye_points_coordinates = self.coordinate_finder.capture_points_coordinates(face_results, "right_eye_points", self.coordinate_finder.face_points["right_eye_points"])
                right_eyebrow_points_coordinates = self.coordinate_finder.capture_points_coordinates(face_results, "right_eyebrow_points", self.coordinate_finder.face_points["right_eyebrow_points"])
                origin_points_coordinates = self.coordinate_finder.capture_points_coordinates(face_results, "origin_points", self.coordinate_finder.face_points["origin_points"])
                
                # DESCRIPTION: Add face landmarks to the CoordinateMathOps class.
                self.coordinate_math_ops.lips_points_coordinates = lips_points_coordinates
                self.coordinate_math_ops.left_eye_points_coordinates = left_eye_points_coordinates
                self.coordinate_math_ops.left_eyebrow_points_coordinates = left_eyebrow_points_coordinates
                self.coordinate_math_ops.right_eye_points_coordinates = right_eye_points_coordinates
                self.coordinate_math_ops.right_eyebrow_points_coordinates = right_eyebrow_points_coordinates
                self.coordinate_math_ops.origin_points_coordinates = origin_points_coordinates
                
                # DESCRIPTION: Calculate other face features using face landmarks.
                eyes_openness = self.coordinate_math_ops.calculate_eye_openness()
                lips_openness = self.coordinate_math_ops.calculate_lips_openness()
                eyebrow_eye_distance = self.coordinate_math_ops.calculate_eyebrow_eye_distance()
                eyebrow_slope = self.coordinate_math_ops.calculate_eyebrow_slope()
                lips_slope = self.coordinate_math_ops.calculate_lips_slope()
                
                # DESCRIPTION: Save face landmarks to a JSON file.
                self.json_utils.save_captured_data(eyes_openness + \
                                                   lips_openness + \
                                                   eyebrow_eye_distance + \
                                                   eyebrow_slope + \
                                                   lips_slope + \
                                                   lips_points_coordinates + \
                                                   left_eye_points_coordinates + \
                                                   left_eyebrow_points_coordinates + \
                                                   right_eye_points_coordinates + \
                                                   right_eyebrow_points_coordinates, 
                                                   self.tfl_data, 
                                                   "Face Landmarks", 
                                                   self.tfl_counter, 
                                                   "data/raw/not_yawning.json")
                
                # DESCRIPTION: Increment the counter for the next face landmarks.
                self.tfl_counter += 1

            # DESCRIPTION: Display the image with face landmarks.
            if yawning == 1:
                text = "Yawning"
            else:
                text = "Not Yawning"

            cv2.putText(image, text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.imshow('Face Mesh', image)

            # DESCRIPTION: If the 'q' key is pressed, break the loop for closing the application.
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # DESCRIPTION: Release the camera (close the camera) and destroy all OpenCV windows.
        self.camera_utils.release_camera()
        cv2.destroyAllWindows()