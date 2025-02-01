# src/data_preprocessing.py
import pandas as pd
import numpy as np


class DataPreprocessor:
    def __init__(self, file_path):
        self.file_path = file_path
        self.data = None
        self.target = None
        self.features = None

    def load_data(self):
        """Load data from the CSV file"""
        self.data = pd.read_csv(self.file_path)

    def preprocess_data(self):
        """Preprocess the data by sepatating features and target and scaling the data."""
        self.features = self.data.drop("medv", axis=1)  # Drop the target column
        self.target = self.data["medv"]  # Targiet is the 'medv' column (House price)

        # Apply Z-score scalling (standardization) to the features using Numpy
        self.features = np.array(self.features, dtype=np.float64)
        self.target = np.array(self.target, dtype=np.float64)

        means = np.mean(self.features, axis=0)
        std_devs = np.std(self.features, axis=0)

        self.features = (self.features - means) / (std_devs + 1e-8)
        return self.features, self.target

    def split_data(self, test_size=0.2, random_state=42):
        self.preprocess_data()
        np.random.seed(random_state)
        indices = np.random.permutation(len(self.features))
        test_size = int(len(self.features) * test_size)

        x_train = self.features[indices[:-test_size]]
        x_val = self.features[indices[-test_size:]]
        y_train = self.target[indices[:-test_size]]
        y_val = self.target[indices[-test_size:]]
        return x_train, x_val, y_train, y_val


if __name__ == "__main__":
    print("OK")
    path = "../data/BostonHousing.csv"
    pr = DataPreprocessor(path)
    pr.load_data()
    x_train, x_val, y_train, y_val = pr.split_data(test_size=0.2, random_state=42)
    print("train", x_train.shape, y_train.shape)
    print("validation", x_val.shape, y_val.shape)
