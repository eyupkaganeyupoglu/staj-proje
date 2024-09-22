import json
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import MinMaxScaler
import pickle
import numpy as np

# DESCRIPTION: Load data from a JSON file and extract features and labels.
def load_data(file_path, label):
    with open(file_path, 'r') as f:
        data = json.load(f)
    features = []
    labels = []
    for item in data:
        for landmark in item["landmarks"]:
            for landmark_key, landmark_value in landmark.items():
                if landmark_key not in ["index", "part"]:
                    # DESCRIPTION: Extract features and labels from the data.
                    features.append(landmark_value)
                    # TODO: Sayısal bir değer mi yoksa string bir label mı olmalı bilmiyorum.
                    labels.append(label)
    return features, labels

# DESCRIPTION: Preprocess the data by scaling the features using Min-Max Scaler.
def preprocess_data(features, labels):
    scaler = MinMaxScaler()
    features = scaler.fit_transform(np.array(features).reshape(-1, 1))
    return features, labels

# DESCRIPTION: Train a Support Vector Machine (SVM) model on the preprocessed data.
def train_model(features, labels):
    # DESCRIPTION: Split the data into training and testing sets.
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

    # DESCRIPTION: Create an SVM model with a radial basis function (RBF) kernel.
    model = svm.SVC(kernel='rbf', gamma='scale')

    # DESCRIPTION: Train the SVM model on the training data.
    # TODO: Loop gerekli olabilir.
    model.fit(X_train, y_train)

    # DESCRIPTION: Make predictions on the testing data.
    # TODO: Loop gerekli olabilir.
    y_pred = model.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    # DESCRIPTION: Save the trained model to a file.
    with open('trained_model.pkl', 'wb') as f:
        pickle.dump(model, f)

    return model

# DESCRIPTION: Test the trained model on a new dataset.
def test_model(model):
    # DESCRIPTION: Load the test data from a JSON file.
    test_features_yawn, test_labels_yawn = load_data('data/raw/yawning.json', 1)
    test_features_not_yawn, test_labels_not_yawn = load_data('data/raw/not_yawning.json', 0)

    # DESCRIPTION: Combine the test data into a single dataset.
    test_features = test_features_yawn + test_features_not_yawn
    test_labels = test_labels_yawn + test_labels_not_yawn

    # DESCRIPTION: Preprocess the test data.
    test_features, test_labels = preprocess_data(test_features, test_labels)

    # DESCRIPTION: Make predictions on the test data.
    # TODO: Loop gerekli olabilir.
    test_pred = model.predict(test_features)
    print("Test Accuracy:", accuracy_score(test_labels, test_pred))
    print("Test Classification Report:")
    print(classification_report(test_labels, test_pred))
    print("Test Confusion Matrix:")
    print(confusion_matrix(test_labels, test_pred))

def main():
    # DESCRIPTION: Load the training data from the JSON files.
    features_yawn, labels_yawn = load_data('data/raw/yawning.json', 1)
    features_not_yawn, labels_not_yawn = load_data('data/raw/not_yawning.json', 0)

    features = features_yawn + features_not_yawn
    labels = labels_yawn + labels_not_yawn

    features, labels = preprocess_data(features, labels)

    model = train_model(features, labels)
    test_model(model)

if __name__ == '__main__':
    main()