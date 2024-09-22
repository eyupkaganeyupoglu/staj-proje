from math import atan, degrees
# ATTENTION: Farklı insanlarda farklı anlamsal ilişki min-max normalizasyonu olabilir. Bu yüzden bu uygulama şimdilik herkes üzerinde max doğruluk oranı yakalayamayabilir.
class CoordinateMathOps:
# ATTENTION: Yüzün Z ekseninde dönmesi durumunda eğim değişiyor. Bu yüzden bu uygulama şimdilik yüz düz kabul edilerek yapılmıştır.
    def __init__(self, lips_points_coordinates, left_eye_points_coordinates, left_eyebrow_points_coordinates, right_eye_points_coordinates, right_eyebrow_points_coordinates, origin_points_coordinates):
        self.lips_points_coordinates = lips_points_coordinates
        self.left_eye_points_coordinates = left_eye_points_coordinates
        self.left_eyebrow_points_coordinates = left_eyebrow_points_coordinates
        self.right_eye_points_coordinates = right_eye_points_coordinates
        self.right_eyebrow_points_coordinates = right_eyebrow_points_coordinates
        self.origin_points_coordinates = origin_points_coordinates
        
    # DESCRIPTION: Calculate the Euclidean distance between two points.
    def calculate_euclidean_distance(self, point1, point2):
        if point1 and point2:
            x1, y1, z1 = point1["x"], point1["y"], point1["z"]
            x2, y2, z2 = point2["x"], point2["y"], point2["z"]
            # DESCRIPTION: Euclidean distance between two points mathematically.
            return ((x2 - x1) ** 2 + (y2 - y1) ** 2 + (z2 - z1) ** 2) ** 0.5
        else:
            return None
    
    # DESCRIPTION: Calculate the slope of the line passing through two points.
    def calculate_2D_slope(self, point1, point2):
        if point1 and point2:
            # DESCRIPTION: Get data for points at indexes 8, 9.
            origin_point_no_8 = next((point for point in self.origin_points_coordinates if point["index"] == 8), None)
            origin_point_no_9 = next((point for point in self.origin_points_coordinates if point["index"] == 9), None)

            # DESCRIPTION: Calculate the angle difference between two slopes.
            if origin_point_no_8["x"] == 0 and origin_point_no_8["y"] == 0 and origin_point_no_8["z"] == 0:
                origin_slope = (origin_point_no_9["y"] - origin_point_no_8["y"]) / (origin_point_no_9["x"] - origin_point_no_8["x"])
                point12_slope = (point2["y"] - point1["y"]) / (point2["x"] - point1["x"])
                return atan((point12_slope - origin_slope) / (1 + point12_slope * origin_slope))
            else:
                print("Origin point 8 is not zero.")
                return None
    
        # 159 (left eye, top), 145 (left eye, bottom), 385 (right eye, top), 380 (right eye, bottom)
        # DESCRIPTION: Calculate the openness of the eyes.
    def calculate_eye_openness(self):
        # DESCRIPTION: Get the data for the points at indexes 159, 145, 385, 380.
        point_159 = next((point for point in self.left_eye_points_coordinates if point["index"] == 159), None)
        point_154 = next((point for point in self.left_eye_points_coordinates if point["index"] == 145), None)
        point_385 = next((point for point in self.right_eye_points_coordinates if point["index"] == 385), None)
        point_380 = next((point for point in self.right_eye_points_coordinates if point["index"] == 380), None)

        # DESCRIPTION: Calculate the Euclidean distance between the top and bottom points of the eyes.
        left_eye_openness = self.calculate_euclidean_distance(point_159, point_154)
        right_eye_openness = self.calculate_euclidean_distance(point_385, point_380)

        # DESCRIPTION: Return the openness of the eyes.
        eyes_openness = [{"part": "eye_openness",
                          "left_eye_openness": left_eye_openness,
                          "right_eye_openness": right_eye_openness}]
        return eyes_openness

    # 13 (lips, center, top), 14 (lips, center, bottom)
    # DESCRIPTION: Calculate the openness of the lips.
    def calculate_lips_openness(self):
        # DESCRIPTION: Get the data for the points at indexes 13 14.
        point_13 = next((point for point in self.lips_points_coordinates if point["index"] == 13), None)
        point_14 = next((point for point in self.lips_points_coordinates if point["index"] == 14), None)

        # DESCRIPTION: Calculate the Euclidean distance between the top and bottom points of the lips.
        lips_openness = self.calculate_euclidean_distance(point_13, point_14)

        # DESCRIPTION: Return the openness of the lips.
        lips_openness = [{"part": "lips_openness",
                          "lips_openness": lips_openness}]
        return lips_openness

    # 52-66 (left eyebrow, center), 159 (left eye, top), 296-282 (right eyebrow, center), 385 (right eye, top)
    # DESCRIPTION: Calculate the distance between the eyebrows and eyes.
    def calculate_eyebrow_eye_distance(self):
        # DESCRIPTION: Get the data for the points at indexes 159, 385, 52, 66, 296, 282.
        point_159 = next((point for point in self.left_eye_points_coordinates if point["index"] == 159), None)
        point_385 = next((point for point in self.right_eye_points_coordinates if point["index"] == 385), None)
        point_52 = next((point for point in self.left_eyebrow_points_coordinates if point["index"] == 52), None)
        point_66 = next((point for point in self.left_eyebrow_points_coordinates if point["index"] == 66), None)
        point_296 = next((point for point in self.right_eyebrow_points_coordinates if point["index"] == 296), None)
        point_282 = next((point for point in self.right_eyebrow_points_coordinates if point["index"] == 282), None)

        # DESCRIPTION: Calculate the center points of the eyebrows.
        left_eyebrow_center  = {"x": (point_52["x"] + point_66["x"]) / 2,
                                "y": (point_52["y"] + point_66["y"]) / 2,
                                "z": (point_52["z"] + point_66["z"]) / 2}
        right_eyebrow_center = {"x": (point_296["x"] + point_282["x"]) / 2,
                                "y": (point_296["y"] + point_282["y"]) / 2,
                                "z": (point_296["z"] + point_282["z"]) / 2}
        
        # DESCRIPTION: Calculate the distance between the eyebrows and eyes.
        left_eyebrow_eye_distance = self.calculate_euclidean_distance(left_eyebrow_center, point_159)
        right_eyebrow_eye_distance = self.calculate_euclidean_distance(right_eyebrow_center, point_385)

        # DESCRIPTION: Return the distance between the eyebrows and eyes.
        eyebrow_eye_distance = [{"part": "eyebrow_eye_distance",
                                 "left_eyebrow_eye_distance": left_eyebrow_eye_distance,
                                 "right_eyebrow_eye_distance": right_eyebrow_eye_distance}]
        return eyebrow_eye_distance

    # 55-105, 46-105 (left eyebrow), 285-334, 276-334 (right eyebrow)
    # DESCRIPTION: Calculate the slope of the eyebrows.
    def calculate_eyebrow_slope(self):
        # DESCRIPTION: Get the data for the points at indexes 55, 105, 46, 285, 334, 276.
        point_55 = next((point for point in self.left_eyebrow_points_coordinates if point["index"] == 55), None)
        point_105 = next((point for point in self.left_eyebrow_points_coordinates if point["index"] == 105), None)
        point_46 = next((point for point in self.left_eyebrow_points_coordinates if point["index"] == 46), None)
        point_285 = next((point for point in self.right_eyebrow_points_coordinates if point["index"] == 285), None)
        point_334 = next((point for point in self.right_eyebrow_points_coordinates if point["index"] == 334), None)
        point_276 = next((point for point in self.right_eyebrow_points_coordinates if point["index"] == 276), None)

        # DESCRIPTION: Calculate the slope of the eyebrows.
        left_eyebrow_slope_55_105 = self.calculate_2D_slope(point_55, point_105)
        left_eyebrow_slope_46_105 = self.calculate_2D_slope(point_46, point_105)
        right_eyebrow_slope_285_334 = self.calculate_2D_slope(point_285, point_334)
        right_eyebrow_slope_276_334 = self.calculate_2D_slope(point_276, point_334)

        # DESCRIPTION: Return the slope of the eyebrows.
        eyebrow_slope = [{"part": "eyebrow_slope",
                          "left_eyebrow_slope_55_105": left_eyebrow_slope_55_105,
                          "left_eyebrow_slope_46_105": left_eyebrow_slope_46_105,
                          "right_eyebrow_slope_285_334": right_eyebrow_slope_285_334,
                          "right_eyebrow_slope_276_334": right_eyebrow_slope_276_334}]
        return eyebrow_slope

    # 61 (left lip, corner), 291 (right lip, corner), 14 (lips, center, bottom)
    # DESCRIPTION: Calculate the slope of the lips.
    def calculate_lips_slope(self):
        # DESCRIPTION: Get the data for the points at indexes 61, 291, 14.
        point_61 = next((point for point in self.lips_points_coordinates if point["index"] == 61), None)
        point_291 = next((point for point in self.lips_points_coordinates if point["index"] == 291), None)
        point_14 = next((point for point in self.lips_points_coordinates if point["index"] == 14), None)

        # DESCRIPTION: Calculate the slope of the lips.
        left_lips_slope = self.calculate_2D_slope(point_61, point_14)
        right_lips_slope = self.calculate_2D_slope(point_291, point_14)

        # DESCRIPTION: Return the slope of the lips.
        lips_slope = [{"part": "lips_slope",
                      "left_lips_slope": left_lips_slope,
                      "right_lips_slope": right_lips_slope}]
        return lips_slope