class CoordinateFinder:
    def __init__(self):
        # DESCRIPTION: Define the face points for capturing face landmarks.
        self.face_points = {"lips_points": [0, 13, 14, 17, 37, 39, 40, 61, 78, 80, 81, 82, 84, 87, 88, 91, 95, 146, 178, 181, 185, 191, 267, 269, 270, 291, 308, 310, 311, 312, 314, 317, 318, 321, 324, 375, 402, 405, 409, 415],
                            "left_eye_points": [7, 33, 133, 144, 145, 153, 154, 155, 157, 158, 159, 160, 161, 163, 173, 246],
                            "left_eyebrow_points": [46, 52, 53, 55, 63, 65, 66, 70, 105, 107],
                            "right_eye_points": [249, 263, 362, 373, 374, 380, 381, 382, 384, 385, 386, 387, 388, 390, 398, 466],
                            "right_eyebrow_points": [276, 282, 283, 285, 293, 295, 296, 300, 334, 336],
                            "origin_points": [8, 9]}

    def capture_points_coordinates(self, face_results, points_type, points):
        try:
            # DESCRIPTION: Get the nose tip landmark and its x, y, and z coordinates.
            origin_landmark = face_results.multi_face_landmarks[0].landmark[8]
            origin_x, origin_y, origin_z = origin_landmark.x, origin_landmark.y, origin_landmark.z
            
            points_coordinates = []

            # DESCRIPTION: Calculate the relative x, y, and z coordinates of the face landmarks with respect to the nose tip landmark.
            for landmark_index in points:
                landmark = face_results.multi_face_landmarks[0].landmark[landmark_index]
                relative_x = landmark.x - origin_x
                relative_y = landmark.y - origin_y
                relative_z = landmark.z - origin_z
                points_coordinates.append({"part": points_type,
                                          "index": landmark_index,
                                          "x": relative_x,
                                          "y": relative_y,
                                          "z": relative_z})
            return points_coordinates
        except Exception as e:
            print(f"Nose tip not found: {e}")
        return None

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