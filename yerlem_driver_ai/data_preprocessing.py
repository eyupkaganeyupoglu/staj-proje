from json_utils import JsonUtils
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, Normalizer

class DataPreprocessing:
    def __init__(self):
        self.json_utils = JsonUtils()

    def relabel(self, data_path, new_data_path):
        data = self.json_utils.load_data(data_path)
        new_data = []
        for landmarks in data:
            id = landmarks["id"]
            label = landmarks["label"]
            temp_points_coordinates = []
            for landmark in landmarks["landmarks"]:
                if landmark["part"] == "eye_openness":
                    eyes_openness = {"part": "eye_openness",
                                     f"left_eye_openness": landmark["left_eye_openness"],
                                     f"right_eye_openness": landmark["right_eye_openness"]}
                    temp_points_coordinates.append(eyes_openness)

                elif landmark["part"] == "lips_openness":
                    lips_openness = {"part": "lips_openness",
                                    f"lips_openness": landmark["lips_openness"]}
                    temp_points_coordinates.append(lips_openness)

                elif landmark["part"] == "eyebrow_eye_distance":
                    eyebrow_eye_distance = {"part": "eyebrow_eye_distance",
                                           f"left_eyebrow_eye_distance": landmark["left_eyebrow_eye_distance"],
                                           f"right_eyebrow_eye_distance": landmark["right_eyebrow_eye_distance"]}
                    temp_points_coordinates.append(eyebrow_eye_distance)

                elif landmark["part"] == "eyebrow_slope":
                    eyebrow_slope = {"part": "eyebrow_slope",
                                    f"left_eyebrow_slope_55_105": landmark["left_eyebrow_slope_55_105"],
                                    f"left_eyebrow_slope_46_105": landmark["left_eyebrow_slope_46_105"],
                                    f"right_eyebrow_slope_285_334": landmark["right_eyebrow_slope_285_334"],
                                    f"right_eyebrow_slope_276_334": landmark["right_eyebrow_slope_276_334"]}
                    temp_points_coordinates.append(eyebrow_slope)

                elif landmark["part"] == "lips_slope":
                    lips_slope = {"part": "lips_slope",
                                 f"left_lips_slope": landmark["left_lips_slope"],
                                 f"right_lips_slope": landmark["right_lips_slope"]}
                    temp_points_coordinates.append(lips_slope)

                elif landmark.get('part') in ("lips_points", "left_eye_points", "left_eyebrow_points", "right_eye_points", "right_eyebrow_points"):
                    point_coordinates = {"part": landmark["part"],
                                         "index": landmark["index"],
                                        f"{landmark.get('part')}_{landmark.get('index')}_x": landmark["x"],
                                        f"{landmark.get('part')}_{landmark.get('index')}_y": landmark["y"],
                                        f"{landmark.get('part')}_{landmark.get('index')}_z": landmark["z"]}
                    temp_points_coordinates.append(point_coordinates)

            new_data.append({"id": id, "label": label, "landmarks": temp_points_coordinates})

        self.json_utils.save_data(new_data, new_data_path)

    def json_to_csv(self, data_path, csv_path):
        data = self.json_utils.load_data(data_path)
        new_data = []
        for landmarks in data:
            temp_points_coordinates = {}
            for landmark in landmarks["landmarks"]:
                if landmark["part"] == "eye_openness":
                    for i in landmark:
                        temp_points_coordinates.update({f"left_eye_openness": landmark["left_eye_openness"],
                                                        f"right_eye_openness": landmark["right_eye_openness"]})

                elif landmark["part"] == "lips_openness":
                    for i in landmark:
                        temp_points_coordinates.update({f"lips_openness": landmark["lips_openness"]})

                elif landmark["part"] == "eyebrow_eye_distance":
                    for i in landmark:
                        temp_points_coordinates.update({f"left_eyebrow_eye_distance": landmark["left_eyebrow_eye_distance"],
                                                        f"right_eyebrow_eye_distance": landmark["right_eyebrow_eye_distance"]})

                elif landmark["part"] == "eyebrow_slope":
                    for i in landmark:
                        temp_points_coordinates.update({f"left_eyebrow_slope_55_105": landmark["left_eyebrow_slope_55_105"],
                                                        f"left_eyebrow_slope_46_105": landmark["left_eyebrow_slope_46_105"],
                                                        f"right_eyebrow_slope_285_334": landmark["right_eyebrow_slope_285_334"],
                                                        f"right_eyebrow_slope_276_334": landmark["right_eyebrow_slope_276_334"]})

                elif landmark["part"] == "lips_slope":
                    for i in landmark:
                        temp_points_coordinates.update({f"left_lips_slope": landmark["left_lips_slope"],
                                                        f"right_lips_slope": landmark["right_lips_slope"]})

                elif landmark.get('part') in ("lips_points", "left_eye_points", "left_eyebrow_points", "right_eye_points", "right_eyebrow_points"):
                    for i in landmark:
                        temp_points_coordinates.update({f"{landmark.get('part')}_{landmark.get('index')}_x": landmark[f"{landmark.get('part')}_{landmark.get('index')}_x"],
                                                        f"{landmark.get('part')}_{landmark.get('index')}_y": landmark[f"{landmark.get('part')}_{landmark.get('index')}_x"],
                                                        f"{landmark.get('part')}_{landmark.get('index')}_z": landmark[f"{landmark.get('part')}_{landmark.get('index')}_x"]})
                        
            new_data.append(temp_points_coordinates)

        df = pd.DataFrame(new_data)
        df.to_csv(csv_path, index=False)

# if __name__ == "__main__":
#     relabeling = DataPreprocessing()
#     relabeling.relabel("data/raw/yawning.json", "data/raw/yawning_relabeled.json")
#     relabeling.relabel("data/raw/not_yawning.json", "data/raw/not_yawning_relabeled.json")
#     relabeling.json_to_csv("data/raw/yawning_relabeled.json", "data/raw/yawning_relabeled.csv")
#     relabeling.json_to_csv("data/raw/not_yawning_relabeled.json", "data/raw/not_yawning_relabeled.csv")