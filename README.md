# Boston House Price Prediction

This prooject aims to prdict house prices in Boston using basic machine learning techniques(linear regression), without using any machine learning libraries like Sikit-learn. We will use fundamental mathematical concept and algorithms to build a predictive model.

## project Structure
- 'data/': Contains the dataset (.CSV).
- 'nootbook/': Contains documentation or jupyter Nootbooks.
- 'Src/': Contains the source code (data preprocessing, etc.).
- 'output/': Contains the results and saved models.

## Project Overview
### 1. Data preprocessing
The 'Data_preprocessing' class handles data preprocessing tasks, including:
- **Loading data** from a CSV file.
- **Splitting data** into training and validation sets.
- **Standardizing data** (feature scaling) using mean and standard deviation of the training set.

The data file used in this project is the **Boston Housing Prices** dataset ('boston_housing_prices.csv').

## 2. Linear Regression Model
The 'LinearRegression' class implenents a simple linear regression model. The model:
- Initializes with random wights and bias.
- Performs gradient descent to minimize the mean Squared Error (MSE).
- Save the training and validation errors after each epoch.

## 3. Plotting Errors
we store the**training errors** and **validation error** during the training process, and plot both on the same graph to visualize the model's performance over time.

## 4. Requirements
The following libraries are required to run the project:
- 'numpy'
- 'pandas'
- 'matplotlib'
You can install these dependenciesby running:

Pip install -r requirements.txt

