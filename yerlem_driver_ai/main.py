import cv2
from yerlem_driver_ai.camera_utils import CameraUtils
from yerlem_driver_ai.file_operation_utils import FileOperationUtils
from yerlem_driver_ai.mediapipe_utils import MediaPipeUtils
from yerlem_driver_ai.feature_extractor_utils import FeatureExtractor
# from yerlem_driver_ai.drawing_utils import DrawingUtils
from yerlem_driver_ai.models_utils import Models

class Application:
    def __init__(self):
        # DESCRIPTION: Initialize CameraUtils, FileOperationUtils, MediaPipeUtils, FeatureExtractor, and Models classes.
        self.camera_utils = CameraUtils()
        self.file_operation_utils = FileOperationUtils()
        self.mp_utils = MediaPipeUtils()
        self.feature_extractor = FeatureExtractor()
        # self.drawing_utils = DrawingUtils(self.mp_utils.mp_drawing, self.mp_utils.mp_drawing_styles, self.mp_utils.mp_face_mesh) #, self.mp_utils.mp_hands
        self.models = Models()

        # DESCRIPTION: Load the Random Forest model for yawning detection.
        self.model_RF = self.models.import_model_joblib("random_forest_model_for_yawning_detection")

        # DESCRIPTION: Load the data from the CSV file for creating a dataset. 
        self.tfl_data = self.file_operation_utils.load_data("data/raw/deneme.csv")

    def run(self):
        while True:
            # DESCRIPTION: Read the frame from the camera.
            image = self.camera_utils.read_frame()
            # DESCRIPTION: Process the frame to detect face landmarks by using MediaPipe Face Mesh.
            face_results = self.mp_utils.process(image)

            # DESCRIPTION: Extract features from the face landmarks.
            features = self.feature_extractor.capture_points_coordinates(face_results)

            # DESCRIPTION: Dataset creation.
            if cv2.waitKey(1) & 0xFF == ord('c'):
                self.file_operation_utils.save_data(features, "data/raw/deneme.csv")
            # DESCRIPTION: Exit the application.
            elif cv2.waitKey(1) & 0xFF == ord('q'):
                break
            # DESCRIPTION: Yawning detection.
            else:
                yawning = self.model_RF.predict(features)
                if yawning == 1:
                    text = "Yawning"
                else:
                    text = "Not Yawning"
            
            # DESCRIPTION: Display the frame with face landmarks.
            # self.drawing_utils.draw_face_landmarks(image, face_results) #, self.coordinate_finder.face_points)

            # DESCRIPTION: Display the text on the frame for yawning detection result.
            cv2.putText(image, text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            # DESCRIPTION: Show the frame.
            cv2.imshow('Face Mesh', image)

        self.camera_utils.release_camera()
        cv2.destroyAllWindows()