from sklearn.ensemble import RandomForestClassifier
# from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib as jb
import pandas as pd

class Models:
    def __init__(self):
        # DESCRIPTION: Load the data from the CSV files for yawning and not yawning.
        self.yawning = pd.read_csv("data/raw/yawning.csv")
        self.not_yawning = pd.read_csv("data/raw/not_yawning.csv")

        # DESCRIPTION: Add the label column to the data.
        self.yawning["label"] = 1
        self.not_yawning["label"] = 0

        # DESCRIPTION: Concatenate the yawning and not yawning data.
        self.data = pd.concat([self.yawning, self.not_yawning], ignore_index=True)

        # DESCRIPTION: Split the data into features and labels.
        self.x = self.data.drop("label", axis=1)
        self.y = self.data["label"]

        # DESCRIPTION: Split the data into training and testing sets.
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.x, self.y, test_size=0.2, random_state=42)

        # DESCRIPTION: Create a Random Forest model.
        self.model_RF = RandomForestClassifier(n_estimators=200)
        # self.scaler_MM = MinMaxScaler()
        # self.scaler_SS = StandardScaler()

    # def preprocess_data(self):
    #     self.x_train = self.scaler_SS.fit_transform(self.x_train)
    #     self.x_test = self.scaler_SS.transform(self.x_test)
    
    # DESCRIPTION: Train the Random Forest model.
    def train_model(self):
        self.model_RF.fit(self.x_train, self.y_train)

    # DESCRIPTION: Evaluate the Random Forest model.
    def model_score(self):
        print(f"Random Forest Model classification report: \n{classification_report(self.y_test, self.model_RF.predict(self.x_test))}")

    # DESCRIPTION: Save the Random Forest model.
    def save_model_joblib(self):
        jb.dump(self.model_RF, "data/models/random_forest_model_for_yawning_detection.joblib")
    
    # DESCRIPTION: Import the Random Forest model.
    def import_model_joblib(self, model_name):
        self.model = jb.load(f"data/models/{model_name}.joblib")
        return self.model

# ATTENTION: If you want to train your model using your own dataset and save it, you can use the following code and run this file alone.
# ATTENTION: Don't forget to define your datasets in the `read_csv`(line: 11, 12) sections and update the `label`(line: 15, 16) and `concat`(line: 19) operations if you are using multiple datasets.
# if __name__ == "__main__":
#     train_model = Models()
#     train_model.train_model()
#     # train_model.preprocess_data()
#     train_model.model_score()
#     train_model.save_model_joblib()