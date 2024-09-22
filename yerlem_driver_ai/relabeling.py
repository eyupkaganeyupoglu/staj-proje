from json_utils import JsonUtils

class Relabeling:
    def __init__(self):
        self.json_utils = JsonUtils()
        self.data = self.json_utils.load_data("data/raw/tired_face_landmarks.json")
        self.new_data = []
        self.temp_points_coordinates = []

    def relabel(self):
        for landmarks in self.data:
            id = landmarks["id"]
            label = landmarks["label"]
            self.temp_points_coordinates = []
            for landmark in landmarks["landmarks"]:
                if landmark["part"] == "eye_openness":
                    eyes_openness = {"part": "eye_openness",
                                     f"{id}_left_eye_openness": landmark["left_eye_openness"],
                                     f"{id}_right_eye_openness": landmark["right_eye_openness"]}
                    self.temp_points_coordinates.append(eyes_openness)

                elif landmark["part"] == "lips_openness":
                    lips_openness = {"part": "lips_openness",
                                    f"{id}_lips_openness": landmark["lips_openness"]}
                    self.temp_points_coordinates.append(lips_openness)

                elif landmark["part"] == "eyebrow_eye_distance":
                    eyebrow_eye_distance = {"part": "eyebrow_eye_distance",
                                           f"{id}_left_eyebrow_eye_distance": landmark["left_eyebrow_eye_distance"],
                                           f"{id}_right_eyebrow_eye_distance": landmark["right_eyebrow_eye_distance"]}
                    self.temp_points_coordinates.append(eyebrow_eye_distance)

                elif landmark["part"] == "eyebrow_slope":
                    eyebrow_slope = {"part": "eyebrow_slope",
                                    f"{id}_left_eyebrow_slope_55_105": landmark["left_eyebrow_slope_55_105"],
                                    f"{id}_left_eyebrow_slope_46_105": landmark["left_eyebrow_slope_46_105"],
                                    f"{id}_right_eyebrow_slope_285_334": landmark["right_eyebrow_slope_285_334"],
                                    f"{id}_right_eyebrow_slope_276_334": landmark["right_eyebrow_slope_276_334"]}
                    self.temp_points_coordinates.append(eyebrow_slope)

                elif landmark["part"] == "lips_slope":
                    lips_slope = {"part": "lips_slope",
                                 f"{id}_left_lips_slope": landmark["left_lips_slope"],
                                 f"{id}_right_lips_slope": landmark["right_lips_slope"]}
                    self.temp_points_coordinates.append(lips_slope)

                elif landmark.get('part') in ("lips_points", "left_eye_points", "left_eyebrow_points", "right_eye_points", "right_eyebrow_points"):
                    point_coordinates = {"part": landmark["part"],
                                         "index": landmark["index"],
                                        f"{id}_{landmark.get('part')}_{landmark.get('index')}_x": landmark["x"],
                                        f"{id}_{landmark.get('part')}_{landmark.get('index')}_y": landmark["y"],
                                        f"{id}_{landmark.get('part')}_{landmark.get('index')}_z": landmark["z"]}
                    self.temp_points_coordinates.append(point_coordinates)

            # TODO: Bir şekilde bütün id'ler en son landmarks listesindeki id oluyor. Yani son id 100 ise hepsi 100 oluyor.
            self.new_data.append({"id": id, "label": label, "landmarks": self.temp_points_coordinates})
        print("breakpoint")
        self.json_utils.save_data(self.new_data, "data/raw/tired_face_landmarks_relabeled.json")

if __name__ == "__main__":
    relabeling = Relabeling()
    relabeling.relabel()