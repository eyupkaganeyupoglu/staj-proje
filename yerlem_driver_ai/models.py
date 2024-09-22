from sklearn.ensemble import RandomForestClassifier
# from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import pickle
import pandas as pd

class Models:
    def __init__(self):
        self.yawning = pd.read_csv("data/raw/yawning_relabeled.csv")
        self.not_yawning = pd.read_csv("data/raw/not_yawning_relabeled.csv")
        self.yawning["label"] = 1
        self.not_yawning["label"] = 0
        self.data = pd.concat([self.yawning, self.not_yawning], ignore_index=True)
        self.x = self.data.drop("label", axis=1)
        self.y = self.data["label"]
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.x, self.y, test_size=0.2, random_state=42)
        self.model_RF = RandomForestClassifier(n_estimators=200)
        # self.scaler_MM = MinMaxScaler()
        # self.scaler_SS = StandardScaler()

    # def preprocess_data(self):
    #     self.x_train = self.scaler_SS.fit_transform(self.x_train)
    #     self.x_test = self.scaler_SS.transform(self.x_test)
    
    def train_model(self):
        self.model_RF.fit(self.x_train, self.y_train)

    def model_score(self):
        print(f"Random Forest Model classification report: \n{classification_report(self.y_test, self.model_RF.predict(self.x_test))}")

    def save_model(self):
        with open("data/models/random_forest_model_for_yawning_detection.pkl", "wb") as f:
            pickle.dump(self.model_RF, f)

    def import_model(self, model_name):
        with open(f"data/models/{model_name}.pkl", "rb") as f:
            self.model = pickle.load(f)
            return self.model

# if __name__ == "__main__":
#     train_model = TrainModel()
#     train_model.train_model()
#     # train_model.preprocess_data()
#     train_model.model_score()
#     train_model.save_model()