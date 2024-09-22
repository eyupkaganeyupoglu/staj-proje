import pandas as pd

class DataPreprocessing1:
    def __init__(self):
        pass

    def relabel(self, captured_data):
        data = captured_data
        new_data = []
        for landmarks in data:
            id = landmarks["id"]
            label = landmarks["label"]
            temp_points_coordinates = []
            for landmark in landmarks["landmarks"]:
                for landmark_i in landmark:
                    if landmark_i["part"] == "eye_openness":
                        eyes_openness = {"part": "eye_openness",
                                        f"left_eye_openness": landmark_i["left_eye_openness"],
                                        f"right_eye_openness": landmark_i["right_eye_openness"]}
                        temp_points_coordinates.append(eyes_openness)

                    elif landmark_i["part"] == "lips_openness":
                        lips_openness = {"part": "lips_openness",
                                        f"lips_openness": landmark_i["lips_openness"]}
                        temp_points_coordinates.append(lips_openness)

                    elif landmark_i["part"] == "eyebrow_eye_distance":
                        eyebrow_eye_distance = {"part": "eyebrow_eye_distance",
                                            f"left_eyebrow_eye_distance": landmark_i["left_eyebrow_eye_distance"],
                                            f"right_eyebrow_eye_distance": landmark_i["right_eyebrow_eye_distance"]}
                        temp_points_coordinates.append(eyebrow_eye_distance)

                    elif landmark_i["part"] == "eyebrow_slope":
                        eyebrow_slope = {"part": "eyebrow_slope",
                                        f"left_eyebrow_slope_55_105": landmark_i["left_eyebrow_slope_55_105"],
                                        f"left_eyebrow_slope_46_105": landmark_i["left_eyebrow_slope_46_105"],
                                        f"right_eyebrow_slope_285_334": landmark_i["right_eyebrow_slope_285_334"],
                                        f"right_eyebrow_slope_276_334": landmark_i["right_eyebrow_slope_276_334"]}
                        temp_points_coordinates.append(eyebrow_slope)

                    elif landmark_i["part"] == "lips_slope":
                        lips_slope = {"part": "lips_slope",
                                    f"left_lips_slope": landmark_i["left_lips_slope"],
                                    f"right_lips_slope": landmark_i["right_lips_slope"]}
                        temp_points_coordinates.append(lips_slope)

                    elif landmark_i.get('part') in ("lips_points", "left_eye_points", "left_eyebrow_points", "right_eye_points", "right_eyebrow_points"):
                        point_coordinates = {"part": landmark_i["part"],
                                            "index": landmark_i["index"],
                                            f"{landmark_i.get('part')}_{landmark_i.get('index')}_x": landmark_i["x"],
                                            f"{landmark_i.get('part')}_{landmark_i.get('index')}_y": landmark_i["y"],
                                            f"{landmark_i.get('part')}_{landmark_i.get('index')}_z": landmark_i["z"]}
                        temp_points_coordinates.append(point_coordinates)

            new_data.append({"id": id, "label": label, "landmarks": temp_points_coordinates})

        return new_data

    def json_to_csv(self, relabeled_data):
        data = relabeled_data
        new_data = []
        for landmarks in data:
            temp_points_coordinates = {}
            for landmark in landmarks["landmarks"]:
                if landmark["part"] == "eye_openness":
                    temp_points_coordinates.update({f"left_eye_openness": landmark["left_eye_openness"],
                                                    f"right_eye_openness": landmark["right_eye_openness"]})

                elif landmark["part"] == "lips_openness":
                    temp_points_coordinates.update({f"lips_openness": landmark["lips_openness"]})

                elif landmark["part"] == "eyebrow_eye_distance":
                    temp_points_coordinates.update({f"left_eyebrow_eye_distance": landmark["left_eyebrow_eye_distance"],
                                                    f"right_eyebrow_eye_distance": landmark["right_eyebrow_eye_distance"]})

                elif landmark["part"] == "eyebrow_slope":
                    temp_points_coordinates.update({f"left_eyebrow_slope_55_105": landmark["left_eyebrow_slope_55_105"],
                                                    f"left_eyebrow_slope_46_105": landmark["left_eyebrow_slope_46_105"],
                                                    f"right_eyebrow_slope_285_334": landmark["right_eyebrow_slope_285_334"],
                                                    f"right_eyebrow_slope_276_334": landmark["right_eyebrow_slope_276_334"]})

                elif landmark["part"] == "lips_slope":
                        temp_points_coordinates.update({f"left_lips_slope": landmark["left_lips_slope"],
                                                        f"right_lips_slope": landmark["right_lips_slope"]})

                elif landmark.get('part') in ("lips_points", "left_eye_points", "left_eyebrow_points", "right_eye_points", "right_eyebrow_points"):
                    temp_points_coordinates.update({f"{landmark.get('part')}_{landmark.get('index')}_x": landmark[f"{landmark.get('part')}_{landmark.get('index')}_x"],
                                                    f"{landmark.get('part')}_{landmark.get('index')}_y": landmark[f"{landmark.get('part')}_{landmark.get('index')}_x"],
                                                    f"{landmark.get('part')}_{landmark.get('index')}_z": landmark[f"{landmark.get('part')}_{landmark.get('index')}_x"]})
                        
            new_data.append(temp_points_coordinates)

        return pd.DataFrame(new_data)
    
# if __name__ == "__main__":
#     relabeling = DataPreprocessing()
#     relabeling.relabel("data/raw/yawning.json", "data/raw/yawning_relabeled.json")
#     relabeling.relabel("data/raw/not_yawning.json", "data/raw/not_yawning_relabeled.json")
#     relabeling.json_to_csv("data/raw/yawning_relabeled.json", "data/raw/yawning_relabeled.csv")
#     relabeling.json_to_csv("data/raw/not_yawning_relabeled.json", "data/raw/not_yawning_relabeled.csv")