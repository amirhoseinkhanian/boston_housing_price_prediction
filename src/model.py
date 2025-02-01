import numpy as np


class LinearRegression:
    def __init__(self, learning_rate=0.01, epochs=200):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = None
        self.bias = None
        self.train_errors = []
        self.validation_errors = []

    def fit(self, x_train, x_val, y_trian, y_val):
        """Train the model using Gradient Descent"""
        num_samples, num_featurs = x_train.shape
        # Initialize weights and bias to random
        self.weights = np.random.randn(num_featurs)
        self.bias = 0
        for epoch in range(self.epochs):
            train_prediction = self.predict(x_train)
            train_error = train_prediction - y_trian

            # Compute gradients for wights and bias
            dW = (1 / num_samples) * np.dot(x_train.T, train_error)
            dB = (1 / num_samples) * np.sum(train_error)

            # Update weights and bias using the gradiants and learning rate
            self.weights -= self.learning_rate * dW
            self.bias -= self.learning_rate * dB
            train_mse = np.mean(train_error**2)
            self.train_errors.append(train_mse)

            val_predictions = self.predict(x_val)
            val_error = val_predictions - y_val
            val_mse = np.mean(val_error**2)
            self.validation_errors.append(val_mse)

            print(
                f"Epoch {epoch+1}/{self.epochs}, Train MSE: {train_mse}, Val MSE: {val_mse}"
            )

    def predict(self, X):
        """Make predictions using the trained model"""
        return np.dot(X, self.weights) + self.bias

    def get_parameters(self):
        """Return the model parameters(weights and bias)"""
        return self.weights, self.bias

    def plot_errors(self):
        """plot errors during train model"""
        import matplotlib.pyplot as plt

        plt.plot(
            range(self.epochs),
            self.train_errors,
            color="blue",
            label="Trianing Error (MSE)",
        )
        plt.plot(
            range(self.epochs),
            self.validation_errors,
            color="orange",
            label="Validation Error (MSE)",
        )
        plt.xlabel("Epochs")
        plt.ylabel("Mean Squared Error (MSE)")
        plt.title("Training and Validation Error Over Epochs")
        plt.legend()
        plt.savefig("../output/Train_validation_error")
        plt.show()
