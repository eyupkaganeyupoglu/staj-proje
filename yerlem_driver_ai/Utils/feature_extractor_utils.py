from math import atan
from pandas import DataFrame

class FeatureExtractor:
    def __init__(self):
        # DESCRIPTION: Define the face points for capturing face landmarks.
        self.face_points = {"lips_points": [0, 13, 14, 17, 37, 39, 40, 61, 78, 80, 81, 82, 84, 87, 88, 91, 95, 146, 178, 181, 185, 191, 267, 269, 270, 291, 308, 310, 311, 312, 314, 317, 318, 321, 324, 375, 402, 405, 409, 415],
                            "left_eye_points": [7, 33, 133, 144, 145, 153, 154, 155, 157, 158, 159, 160, 161, 163, 173, 246],
                            "left_eyebrow_points": [46, 52, 53, 55, 63, 65, 66, 70, 105, 107],
                            "right_eye_points": [249, 263, 362, 373, 374, 380, 381, 382, 384, 385, 386, 387, 388, 390, 398, 466],
                            "right_eyebrow_points": [276, 282, 283, 285, 293, 295, 296, 300, 334, 336]}

    # DESCRIPTION: Capture coordinates and features of the face landmarks.
    def capture_points_coordinates(self, face_results):
        try:
            # DESCRIPTION: Determination of points (8, 9) to be used in the origin and slope of the face.
            origin_point_no_8 = face_results.multi_face_landmarks[0].landmark[8]
            origin_point_no_9 = face_results.multi_face_landmarks[0].landmark[9]

            # DESCRIPTION: Define the indices of the face landmarks for specific features.
            feature_landmark_indices = [159, 145, 386, 374, 13, 14, 52, 66, 296, 282, 55, 46, 105, 285, 276, 334, 61, 291]
            # DESCRIPTION: Initialize the dictionary for storing the coordinates of the face landmarks for specific features.
            points_dict = {}
            # DESCRIPTION: Initialize the dictionary for storing the coordinates of the face landmarks.
            points_coordinates = {}

            # DESCRIPTION: Capture the coordinates of the face landmarks.
            for face_point in self.face_points:
                for landmark_index in self.face_points[face_point]:
                    # DESCRIPTION: Get the face landmark and its x, y, and z coordinates.
                    landmark = face_results.multi_face_landmarks[0].landmark[landmark_index]
                    # DESCRIPTION: Calculate the relative x, y, and z coordinates of the face landmarks with respect to the origin point.
                    relative_x = landmark.x - origin_point_no_8.x
                    relative_y = landmark.y - origin_point_no_8.y
                    relative_z = landmark.z - origin_point_no_8.z
                    # DESCRIPTION: Store the relative x, y, and z coordinates of the face landmarks.
                    # ATTENTION: "face_point" is of type <str> and refers to the key in face_points[face_point].
                    # ATTENTION: Example, face_points["lips_points"] = [0, 13, 14, ...] but face_point = "lips_points".
                    points_coordinates.update({f"{face_point}_{landmark_index}_x": relative_x,
                                               f"{face_point}_{landmark_index}_y": relative_y,
                                               f"{face_point}_{landmark_index}_z": relative_z})
                    
                    # DESCRIPTION: Store the relative x, y, and z coordinates of the face landmarks for specific features.
                    if landmark_index in feature_landmark_indices:
                        points_dict[landmark_index] = {"x": relative_x, "y": relative_y, "z": relative_z}

            # DESCRIPTION: Get the face landmarks for specific features.
            point_159 = points_dict.get(159)
            point_145 = points_dict.get(145)
            point_386 = points_dict.get(386)
            point_374 = points_dict.get(374)
            point_13 = points_dict.get(13)
            point_14 = points_dict.get(14)
            point_52 = points_dict.get(52)
            point_66 = points_dict.get(66)
            point_296 = points_dict.get(296)
            point_282 = points_dict.get(282)
            point_55 = points_dict.get(55)
            point_46 = points_dict.get(46)
            point_105 = points_dict.get(105)
            point_285 = points_dict.get(285)
            point_276 = points_dict.get(276)
            point_334 = points_dict.get(334)
            point_61 = points_dict.get(61)
            point_291 = points_dict.get(291)
            
            # DESCRIPTION: Calculate the features of the face landmarks.
            points_coordinates.update(self.calculate_eye_openness(point_159, point_145, point_386, point_374))
            points_coordinates.update(self.calculate_lips_openness(point_13, point_14))
            points_coordinates.update(self.calculate_eyebrow_eye_distance(point_159, point_386, point_52, point_66, point_296, point_282))
            points_coordinates.update(self.calculate_eyebrow_slope(point_55, point_105, point_46, point_285, point_334, point_276, origin_point_no_8, origin_point_no_9))
            points_coordinates.update(self.calculate_lips_slope(point_61, point_291, point_14, origin_point_no_8, origin_point_no_9))
            
            # DESCRIPTION: Return the DataFrame of the face landmarks.
            return DataFrame([points_coordinates])
        except Exception as e:
            print(f"Nose tip not found: {e}")
            return None
    
    # 159 (left eyelid, top middle)
    # 145 (left eyelid, bottom middle)
    # 386 (right eyelid, top middle)
    # 374 (right eyelid, bottom middle)
    # 13 (lips, center, top)
    # 14 (lips, center, bottom)
    # 52 (left eyebrow, middle bottom left point)
    # 66 (middle top right eyebrow)
    # 296 (middle top left eyebrow)
    # 282 (middle bottom right eyebrow)
    # 55 (right bottom eyebrow)
    # 46 (left bottom eyebrow)
    # 105 (middle top eyebrow)
    # 285 (right bottom eyebrow)
    # 276 (left bottom eyebrow)
    # 334 (middle right top eyebrow)
    # 61 (left lip, corner)
    # 291 (right lip, corner)

    # DESCRIPTION: Calculate the Euclidean distance between two points.
    def calculate_euclidean_distance(self, point1, point2):
        if point1 and point2:
            x1, y1, z1 = point1["x"], point1["y"], point1["z"]
            x2, y2, z2 = point2["x"], point2["y"], point2["z"]

            # DESCRIPTION: Euclidean distance between two points mathematically.
            return ((x2 - x1) ** 2 + (y2 - y1) ** 2 + (z2 - z1) ** 2) ** 0.5
        else:
            return None
    
    # DESCRIPTION: Calculate the slope between two points.
    def calculate_2D_slope(self, point1, point2, origin_point_no_8, origin_point_no_9):
        if point1 and point2:
            # DESCRIPTION: Calculate the angle difference between two slopes.
            origin_slope = (origin_point_no_9.y - origin_point_no_8.y) / (origin_point_no_9.x - origin_point_no_8.x)
            point12_slope = (point2["y"] - point1["y"]) / (point2["x"] - point1["x"])

            # DESCRIPTION: Calculate the slope between two points by trigonometric subtraction formula.
            return atan((point12_slope - origin_slope) / (1 + point12_slope * origin_slope))
    
    # DESCRIPTION: Calculate eye openness.
    def calculate_eye_openness(self, point_159, point_145, point_386, point_374):
        # DESCRIPTION: Calculate the Euclidean distance between the top and bottom points of the eyes.
        left_eye_openness = self.calculate_euclidean_distance(point_159, point_145)
        right_eye_openness = self.calculate_euclidean_distance(point_386, point_374)

        # DESCRIPTION: Return the openness of the eyes.
        eyes_openness = {"left_eye_openness": left_eye_openness,
                         "right_eye_openness": right_eye_openness}
        return eyes_openness
    
    # DESCRIPTION: Calculate lips openness.
    def calculate_lips_openness(self, point_13, point_14):
        # DESCRIPTION: Calculate the Euclidean distance between the top and bottom points of the lips.
        lips_openness = self.calculate_euclidean_distance(point_13, point_14)

        # DESCRIPTION: Return the openness of the lips.
        lips_openness = {"lips_openness": lips_openness}
        return lips_openness
    
    # DESCRIPTION: Calculate the distance between the eyebrows and eyes.
    def calculate_eyebrow_eye_distance(self, point_159, point_386, point_52, point_66, point_296, point_282):
        # DESCRIPTION: Calculate the center points of the eyebrows.
        left_eyebrow_center  = {"x": (point_52["x"] + point_66["x"]) / 2,
                                "y": (point_52["y"] + point_66["y"]) / 2,
                                "z": (point_52["z"] + point_66["z"]) / 2}
        right_eyebrow_center = {"x": (point_296["x"] + point_282["x"]) / 2,
                                "y": (point_296["y"] + point_282["y"]) / 2,
                                "z": (point_296["z"] + point_282["z"]) / 2}
        
        # DESCRIPTION: Calculate the distance between the eyebrows and eyes.
        left_eyebrow_eye_distance = self.calculate_euclidean_distance(left_eyebrow_center, point_159)
        right_eyebrow_eye_distance = self.calculate_euclidean_distance(right_eyebrow_center, point_386)

        # DESCRIPTION: Return the distance between the eyebrows and eyes.
        eyebrow_eye_distance = {"left_eyebrow_eye_distance": left_eyebrow_eye_distance,
                                "right_eyebrow_eye_distance": right_eyebrow_eye_distance}
        return eyebrow_eye_distance
    
    # DESCRIPTION: Calculate the slope of the eyebrows.
    def calculate_eyebrow_slope(self, point_55, point_105, point_46, point_285, point_334, point_276, origin_point_no_8, origin_point_no_9):
        # DESCRIPTION: Calculate the slope of the eyebrows.
        left_eyebrow_slope_55_105 = self.calculate_2D_slope(point_55, point_105, origin_point_no_8, origin_point_no_9)
        left_eyebrow_slope_46_105 = self.calculate_2D_slope(point_46, point_105, origin_point_no_8, origin_point_no_9)
        right_eyebrow_slope_285_334 = self.calculate_2D_slope(point_285, point_334, origin_point_no_8, origin_point_no_9)
        right_eyebrow_slope_276_334 = self.calculate_2D_slope(point_276, point_334, origin_point_no_8, origin_point_no_9)

        # DESCRIPTION: Return the slope of the eyebrows.
        eyebrow_slope = {"left_eyebrow_slope_55_105": left_eyebrow_slope_55_105,
                         "left_eyebrow_slope_46_105": left_eyebrow_slope_46_105,
                         "right_eyebrow_slope_285_334": right_eyebrow_slope_285_334,
                         "right_eyebrow_slope_276_334": right_eyebrow_slope_276_334}
        return eyebrow_slope
    
    # DESCRIPTION: Calculate the slope of the lips.
    def calculate_lips_slope(self, point_61, point_291, point_14, origin_point_no_8, origin_point_no_9):
        # DESCRIPTION: Calculate the slope of the lips.
        left_lips_slope = self.calculate_2D_slope(point_61, point_14, origin_point_no_8, origin_point_no_9)
        right_lips_slope = self.calculate_2D_slope(point_291, point_14, origin_point_no_8, origin_point_no_9)

        # DESCRIPTION: Return the slope of the lips.
        lips_slope = {"left_lips_slope": left_lips_slope,
                      "right_lips_slope": right_lips_slope}
        return lips_slope

    # # TODO: Sağ ve sol eli ayrı ayrı tespit etmenin yolunu bul.
    # def capture_hand_points(self, hand_results, face_results):
    #     try:
    #         # DESCRIPTION: Get the nose tip landmark and its x, y, and z coordinates.
    #         origin_landmark = face_results.multi_face_landmarks[0].landmark[8]
    #         origin_x, origin_y, origin_z = origin_landmark.x, origin_landmark.y, origin_landmark.z
            
    #         hand_points = []

    #         # DESCRIPTION: Calculate the relative x, y, and z coordinates of the hand landmarks with respect to the nose tip landmark.
    #         for landmark in hand_results.multi_hand_landmarks[0].landmark:
    #             relative_x = landmark.x - origin_x
    #             relative_y = landmark.y - origin_y
    #             relative_z = landmark.z - origin_z
    #             hand_points.append({"x": relative_x, "y": relative_y, "z": relative_z})
    #             # hand_points.append({"part": "hand",
    #             #                     "index": landmark_index,
    #             #                     "x": relative_x,
    #             #                     "y": relative_y,
    #             #                     "z": relative_z})
    #         return hand_points
    #     except Exception as e:
    #         print(f"Nose tip not found: {e}")
    #     return None