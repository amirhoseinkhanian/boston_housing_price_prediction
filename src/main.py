import numpy as np
import pandas as pd
from data_preprocessing import DataPreprocessor
from model import LinearRegression


def check_nan(predictions):
    if np.any(np.isnan(predictions)):
        print("Nan detected in predictions!")
    else:
        print("No Nan in predictions.")


def calculate_metrics(y_true, y_pred):
    """Calculate MSE, RMSE"""
    mse = np.mean((y_true - y_pred) ** 2)
    rmse = np.sqrt(mse)
    return mse, rmse


def main():
    # Specify the file path for the data
    file_paht = "../data/BostonHousing.csv"
    # Create an object of the DataPreorocessor class and load the data
    preprocessor = DataPreprocessor(file_paht)
    preprocessor.load_data()

    # Convert the feature and target into numpy array for model input
    preprocessor.preprocess_data()
    x_train, x_val, y_train, y_val = preprocessor.split_data(
        test_size=0.3, random_state=42
    )
    x_train = np.array(x_train, dtype=np.float64)
    y_train = np.array(y_train, dtype=np.float64)
    x_val = np.array(x_val, dtype=np.float64)
    y_val = np.array(y_val, dtype=np.float64)
    # Create an object of the LinearRegression class
    model = LinearRegression(learning_rate=0.01, epochs=1000)
    # Train model
    model.fit(x_train, x_val, y_train, y_val)

    # Make predictions using the trained model
    predictions = model.predict(x_val)
    check_nan(predictions)
    # Calculate MSE, RMSE
    mse, rmse = calculate_metrics(y_val, predictions)

    print(f"MSE: {mse}")
    print(f"RMSE: {rmse}")

    # Display the frist 5 predictions and actual target values
    for i in range(5):
        print(f"Predicted: {predictions[i]}, Actual: {y_val[i]}")

    model.plot_errors()
    out_put = pd.DataFrame(predictions, columns=["predictions"])
    out_put.to_csv("../output/predictions.csv", index=False)


if __name__ == "__main__":
    main()
