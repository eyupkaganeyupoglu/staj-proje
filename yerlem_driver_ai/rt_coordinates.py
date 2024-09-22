from math import atan

class RTCoordinates:
    def __init__(self, lips_points_coordinates, left_eye_points_coordinates, left_eyebrow_points_coordinates, right_eye_points_coordinates, right_eyebrow_points_coordinates, origin_points_coordinates):
        self.face_points = {"lips_points": [0, 13, 14, 17, 37, 39, 40, 61, 78, 80, 81, 82, 84, 87, 88, 91, 95, 146, 178, 181, 185, 191, 267, 269, 270, 291, 308, 310, 311, 312, 314, 317, 318, 321, 324, 375, 402, 405, 409, 415],
                            "left_eye_points": [7, 33, 133, 144, 145, 153, 154, 155, 157, 158, 159, 160, 161, 163, 173, 246],
                            "left_eyebrow_points": [46, 52, 53, 55, 63, 65, 66, 70, 105, 107],
                            "right_eye_points": [249, 263, 362, 373, 374, 380, 381, 382, 384, 385, 386, 387, 388, 390, 398, 466],
                            "right_eyebrow_points": [276, 282, 283, 285, 293, 295, 296, 300, 334, 336],
                            "origin_points": [8, 9]}
        
        self.origin_points_coordinates = origin_points_coordinates
        self.lips_points_coordinates = lips_points_coordinates
        self.left_eye_points_coordinates = left_eye_points_coordinates
        self.left_eyebrow_points_coordinates = left_eyebrow_points_coordinates
        self.right_eye_points_coordinates = right_eye_points_coordinates
        self.right_eyebrow_points_coordinates = right_eyebrow_points_coordinates
    
    # eyes_openness
    # lips_openness
    # eyebrow_eye_distance
    # eyebrow_slope
    # lips_slope
    # lips_points_coordinates
    # left_eye_points_coordinates
    # left_eyebrow_points_coordinates
    # right_eye_points_coordinates
    # right_eyebrow_points_coordinates
        
    def capture_points_coordinates(self, face_results, points_type, points):
        try:
            # DESCRIPTION: Get the nose tip landmark and its x, y, and z coordinates.
            origin_landmark = face_results.multi_face_landmarks[0].landmark[8]
            origin_x, origin_y, origin_z = origin_landmark.x, origin_landmark.y, origin_landmark.z
            
            points_coordinates = {}

            # DESCRIPTION: Calculate the relative x, y, and z coordinates of the face landmarks with respect to the nose tip landmark.
            for landmark_index in points:
                landmark = face_results.multi_face_landmarks[0].landmark[landmark_index]
                relative_x = landmark.x - origin_x
                relative_y = landmark.y - origin_y
                relative_z = landmark.z - origin_z
                points_coordinates.update({f"{points_type}_{landmark_index}_x": relative_x,
                                           f"{points_type}_{landmark_index}_y": relative_y,
                                           f"{points_type}_{landmark_index}_z": relative_z})
            return points_coordinates
        except Exception as e:
            print(f"Nose tip not found: {e}")
        return None
    
    # 159 (left eye, top), 145 (left eye, bottom), 385 (right eye, top), 380 (right eye, bottom)
    # DESCRIPTION: Calculate the openness of the eyes.
    def calculate_eye_openness(self):
        # DESCRIPTION: Get the data for the points at indexes 159, 145, 385, 380.
        point_159 = {"x": self.left_eye_points_coordinates[f"left_eye_points_159_x"],
                     "y": self.left_eye_points_coordinates[f"left_eye_points_159_y"],
                     "z": self.left_eye_points_coordinates[f"left_eye_points_159_z"]}
        point_154 = {"x": self.left_eye_points_coordinates[f"left_eye_points_154_x"],
                     "y": self.left_eye_points_coordinates[f"left_eye_points_154_y"],
                     "z": self.left_eye_points_coordinates[f"left_eye_points_154_z"]}
        point_385 = {"x": self.right_eye_points_coordinates[f"right_eye_points_385_x"],
                     "y": self.right_eye_points_coordinates[f"right_eye_points_385_y"],
                     "z": self.right_eye_points_coordinates[f"right_eye_points_385_z"]}
        point_380 = {"x": self.right_eye_points_coordinates[f"right_eye_points_380_x"],
                     "y": self.right_eye_points_coordinates[f"right_eye_points_380_y"],
                     "z": self.right_eye_points_coordinates[f"right_eye_points_380_z"]}

        # DESCRIPTION: Calculate the Euclidean distance between the top and bottom points of the eyes.
        left_eye_openness = ((point_154["x"] - point_159["x"]) ** 2 + (point_154["y"] - point_159["y"]) ** 2 + (point_154["z"] - point_159["z"]) ** 2) ** 0.5
        right_eye_openness = ((point_380["x"] - point_385["x"]) ** 2 + (point_380["y"] - point_385["y"]) ** 2 + (point_380["z"] - point_385["z"]) ** 2) ** 0.5

        # DESCRIPTION: Return the openness of the eyes.
        eyes_openness = {"left_eye_openness": left_eye_openness,
                         "right_eye_openness": right_eye_openness}
        return eyes_openness
    
    # 13 (lips, center, top), 14 (lips, center, bottom)
    # DESCRIPTION: Calculate the openness of the lips.
    def calculate_lips_openness(self):
        # DESCRIPTION: Get the data for the points at indexes 13 14.
        point_13 = {"x": self.lips_points_coordinates[f"lips_points_13_x"],
                    "y": self.lips_points_coordinates[f"lips_points_13_y"],
                    "z": self.lips_points_coordinates[f"lips_points_13_z"]}
        point_14 = {"x": self.lips_points_coordinates[f"lips_points_14_x"],
                    "y": self.lips_points_coordinates[f"lips_points_14_y"],
                    "z": self.lips_points_coordinates[f"lips_points_14_z"]}

        # DESCRIPTION: Calculate the Euclidean distance between the top and bottom points of the lips.
        lips_openness = ((point_14["x"] - point_13["x"]) ** 2 + (point_14["y"] - point_13["y"]) ** 2 + (point_14["z"] - point_13["z"]) ** 2) ** 0.5

        # DESCRIPTION: Return the openness of the lips.
        lips_openness = {"lips_openness": lips_openness}
        return lips_openness
    
    # 52-66 (left eyebrow, center), 159 (left eye, top), 296-282 (right eyebrow, center), 385 (right eye, top)
    # DESCRIPTION: Calculate the distance between the eyebrows and eyes.
    def calculate_eyebrow_eye_distance(self):
        # DESCRIPTION: Get the data for the points at indexes 159, 385, 52, 66, 296, 282.
        point_159 = {"x": self.left_eye_points_coordinates[f"left_eye_points_159_x"],
                     "y": self.left_eye_points_coordinates[f"left_eye_points_159_y"],
                     "z": self.left_eye_points_coordinates[f"left_eye_points_159_z"]}
        point_385 = {"x": self.right_eye_points_coordinates[f"right_eye_points_385_x"],
                     "y": self.right_eye_points_coordinates[f"right_eye_points_385_y"],
                     "z": self.right_eye_points_coordinates[f"right_eye_points_385_z"]}
        point_52 = {"x": self.left_eyebrow_points_coordinates[f"left_eyebrow_points_52_x"],
                    "y": self.left_eyebrow_points_coordinates[f"left_eyebrow_points_52_y"],
                    "z": self.left_eyebrow_points_coordinates[f"left_eyebrow_points_52_z"]}
        point_66 = {"x": self.left_eyebrow_points_coordinates[f"left_eyebrow_points_66_x"],
                    "y": self.left_eyebrow_points_coordinates[f"left_eyebrow_points_66_y"],
                    "z": self.left_eyebrow_points_coordinates[f"left_eyebrow_points_66_z"]}
        point_296 = {"x": self.right_eyebrow_points_coordinates[f"right_eyebrow_points_296_x"],
                     "y": self.right_eyebrow_points_coordinates[f"right_eyebrow_points_296_y"],
                     "z": self.right_eyebrow_points_coordinates[f"right_eyebrow_points_296_z"]}
        point_282 = {"x": self.right_eyebrow_points_coordinates[f"right_eyebrow_points_282_x"],
                     "y": self.right_eyebrow_points_coordinates[f"right_eyebrow_points_282_y"],
                     "z": self.right_eyebrow_points_coordinates[f"right_eyebrow_points_282_z"]}

        # DESCRIPTION: Calculate the center points of the eyebrows.
        left_eyebrow_center  = {"x": (point_52["x"] + point_66["x"]) / 2,
                                "y": (point_52["y"] + point_66["y"]) / 2,
                                "z": (point_52["z"] + point_66["z"]) / 2}
        right_eyebrow_center = {"x": (point_296["x"] + point_282["x"]) / 2,
                                "y": (point_296["y"] + point_282["y"]) / 2,
                                "z": (point_296["z"] + point_282["z"]) / 2}
        
        # DESCRIPTION: Calculate the distance between the eyebrows and eyes.
        left_eyebrow_eye_distance = ((left_eyebrow_center["x"] - point_159["x"]) ** 2 + (left_eyebrow_center["y"] - point_159["y"]) ** 2 + (left_eyebrow_center["z"] - point_159["z"]) ** 2) ** 0.5
        right_eyebrow_eye_distance = ((right_eyebrow_center["x"] - point_385["x"]) ** 2 + (right_eyebrow_center["y"] - point_385["y"]) ** 2 + (right_eyebrow_center["z"] - point_385["z"]) ** 2) ** 0.5

        # DESCRIPTION: Return the distance between the eyebrows and eyes.
        eyebrow_eye_distance = {"left_eyebrow_eye_distance": left_eyebrow_eye_distance,
                                 "right_eyebrow_eye_distance": right_eyebrow_eye_distance}
        return eyebrow_eye_distance
    
    # 55-105, 46-105 (left eyebrow), 285-334, 276-334 (right eyebrow)
    # DESCRIPTION: Calculate the slope of the eyebrows.
    def calculate_eyebrow_slope(self):
        # DESCRIPTION: Get the data for the points at indexes 55, 105, 46, 285, 334, 276.
        point_55 = {"x": self.left_eyebrow_points_coordinates[f"left_eyebrow_points_55_x"],
                    "y": self.left_eyebrow_points_coordinates[f"left_eyebrow_points_55_y"],
                    "z": self.left_eyebrow_points_coordinates[f"left_eyebrow_points_55_z"]}
        point_105 = {"x": self.left_eyebrow_points_coordinates[f"left_eyebrow_points_105_x"],
                     "y": self.left_eyebrow_points_coordinates[f"left_eyebrow_points_105_y"],
                     "z": self.left_eyebrow_points_coordinates[f"left_eyebrow_points_105_z"]}
        point_46 = {"x": self.left_eyebrow_points_coordinates[f"left_eyebrow_points_46_x"],
                    "y": self.left_eyebrow_points_coordinates[f"left_eyebrow_points_46_y"],
                    "z": self.left_eyebrow_points_coordinates[f"left_eyebrow_points_46_z"]}
        point_285 = {"x": self.right_eyebrow_points_coordinates[f"right_eyebrow_points_285_x"],
                     "y": self.right_eyebrow_points_coordinates[f"right_eyebrow_points_285_y"],
                     "z": self.right_eyebrow_points_coordinates[f"right_eyebrow_points_285_z"]}
        point_334 = {"x": self.right_eyebrow_points_coordinates[f"right_eyebrow_points_334_x"],
                     "y": self.right_eyebrow_points_coordinates[f"right_eyebrow_points_334_y"],
                     "z": self.right_eyebrow_points_coordinates[f"right_eyebrow_points_334_z"]}
        point_276 = {"x": self.right_eyebrow_points_coordinates[f"right_eyebrow_points_276_x"],
                     "y": self.right_eyebrow_points_coordinates[f"right_eyebrow_points_276_y"],
                     "z": self.right_eyebrow_points_coordinates[f"right_eyebrow_points_276_z"]}
        origin_point_no_8 = {"x": self.origin_points_coordinates[f"origin_points_8_x"],
                             "y": self.origin_points_coordinates[f"origin_points_8_y"],
                             "z": self.origin_points_coordinates[f"origin_points_8_z"]}
        origin_point_no_9 = {"x": self.origin_points_coordinates[f"origin_points_9_x"],
                             "y": self.origin_points_coordinates[f"origin_points_9_y"],
                             "z": self.origin_points_coordinates[f"origin_points_9_z"]}
        
        if origin_point_no_8["x"] == 0 and origin_point_no_8["y"] == 0 and origin_point_no_8["z"] == 0:
            origin_slope = (origin_point_no_9["y"] - origin_point_no_8["y"]) / (origin_point_no_9["x"] - origin_point_no_8["x"])
        else:
                print("Origin point 8 is not zero.")
                return None
        left_eyebrow_slope_55_105 = atan(((point_105["y"] - point_55["y"]) / (point_105["x"] - point_55["x"]) - origin_slope) / (1 + ((point_105["y"] - point_55["y"]) / (point_105["x"] - point_55["x"])) * origin_slope))
        left_eyebrow_slope_46_105 = atan(((point_105["y"] - point_46["y"]) / (point_105["x"] - point_46["x"]) - origin_slope) / (1 + ((point_105["y"] - point_46["y"]) / (point_105["x"] - point_46["x"])) * origin_slope))
        right_eyebrow_slope_285_334 = atan(((point_334["y"] - point_285["y"]) / (point_334["x"] - point_285["x"]) - origin_slope) / (1 + ((point_334["y"] - point_285["y"]) / (point_334["x"] - point_285["x"])) * origin_slope))
        right_eyebrow_slope_276_334 = atan(((point_334["y"] - point_276["y"]) / (point_334["x"] - point_276["x"]) - origin_slope) / (1 + ((point_334["y"] - point_276["y"]) / (point_334["x"] - point_276["x"])) * origin_slope))

        # DESCRIPTION: Return the slope of the eyebrows.
        eyebrow_slope = {"left_eyebrow_slope_55_105": left_eyebrow_slope_55_105,
                          "left_eyebrow_slope_46_105": left_eyebrow_slope_46_105,
                          "right_eyebrow_slope_285_334": right_eyebrow_slope_285_334,
                          "right_eyebrow_slope_276_334": right_eyebrow_slope_276_334}
        return eyebrow_slope
    
    # 61 (left lip, corner), 291 (right lip, corner), 14 (lips, center, bottom)
    # DESCRIPTION: Calculate the slope of the lips.
    def calculate_lips_slope(self):
        # DESCRIPTION: Get the data for the points at indexes 61, 291, 14.
        point_61 = {"x": self.lips_points_coordinates[f"lips_points_61_x"],
                    "y": self.lips_points_coordinates[f"lips_points_61_y"],
                    "z": self.lips_points_coordinates[f"lips_points_61_z"]}
        point_291 = {"x": self.lips_points_coordinates[f"lips_points_291_x"],
                     "y": self.lips_points_coordinates[f"lips_points_291_y"],
                     "z": self.lips_points_coordinates[f"lips_points_291_z"]}
        point_14 = {"x": self.lips_points_coordinates[f"lips_points_14_x"],
                    "y": self.lips_points_coordinates[f"lips_points_14_y"],
                    "z": self.lips_points_coordinates[f"lips_points_14_z"]}
        origin_point_no_8 = {"x": self.origin_points_coordinates[f"origin_points_8_x"],
                             "y": self.origin_points_coordinates[f"origin_points_8_y"],
                             "z": self.origin_points_coordinates[f"origin_points_8_z"]}
        origin_point_no_9 = {"x": self.origin_points_coordinates[f"origin_points_9_x"],
                             "y": self.origin_points_coordinates[f"origin_points_9_y"],
                             "z": self.origin_points_coordinates[f"origin_points_9_z"]}
        
        if origin_point_no_8["x"] == 0 and origin_point_no_8["y"] == 0 and origin_point_no_8["z"] == 0:
            origin_slope = (origin_point_no_9["y"] - origin_point_no_8["y"]) / (origin_point_no_9["x"] - origin_point_no_8["x"])
        else:
                print("Origin point 8 is not zero.")
                return None
        
        # DESCRIPTION: Calculate the slope of the lips.
        left_lips_slope = atan(((point_14["y"] - point_61["y"]) / (point_14["x"] - point_61["x"]) - origin_slope) / (1 + ((point_14["y"] - point_61["y"]) / (point_14["x"] - point_61["x"])) * origin_slope))
        right_lips_slope = atan(((point_14["y"] - point_291["y"]) / (point_14["x"] - point_291["x"]) - origin_slope) / (1 + ((point_14["y"] - point_291["y"]) / (point_14["x"] - point_291["x"])) * origin_slope))

        # DESCRIPTION: Return the slope of the lips.
        lips_slope = {"left_lips_slope": left_lips_slope,
                      "right_lips_slope": right_lips_slope}
        return lips_slope