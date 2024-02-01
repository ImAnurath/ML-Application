import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

def standardization(data):
    mean = np.mean(data)
    deviation = np.sqrt(np.mean(data**2) - mean**2)
    standardized = (data - mean) / deviation
    return standardized

def leastSquares(features, label):
    features = np.c_[np.ones(features.shape[0]), features]
    # A == (X^T * X)^-1 * X^T * Y
    trans = features.transpose()
    mult = np.matmul(trans, features)
    inv = np.linalg.inv(mult)
    sec = np.matmul(inv, trans)
    beta = np.matmul(sec, label)
    return beta

def draw(x_features, betas, labels):
    predicted = np.dot(test_intercept_standardized,beta_standardized)
    plt.figure(figsize=(10, 6))
    for month in data['month'].unique():
        month_data = data[data['month'] == month]
        plt.scatter(month_data['day'], month_data['Temperature'], label=f'Month {month}')


    plt.scatter([6, 7, 8, 9], predicted, marker='^', color='pink', label='Predicted')
    plt.plot([6, 7, 8, 9], predicted, color='pink', linestyle='-', linewidth=2)

    plt.title("Temperature Across Months with Predicted Values")
    plt.xlabel('Day')
    plt.ylabel('Temperature')
    plt.legend()
    plt.grid(True)
    plt.show()

# Data Manipulation
script_path = os.path.dirname(os.path.realpath(__file__))
data_name = "Forest_Fire.csv"
data_path = os.path.join(script_path, data_name)
data = pd.read_csv(data_path)
labels = data["Temperature"].values
features = data.iloc[:, 4:-2].values


test_cases = np.array([
                         [87, 29, 0.5, 45.9, 3.5,  7.9,   0.4, 3.4,  0.2],  # 27
                         [65, 14, 0.0, 85.4, 16.0, 44.5,  4.5, 16.9, 6.5],  # 30
                         [56, 14, 0.0, 89.0, 29.4, 115.6, 7.5, 36.0, 15.2], # 35
                         [74, 15, 1.1, 59.5, 4.7,  8.2,   0.8, 4.6,  0.3],  # 29
                         ])

standardized_features = np.apply_along_axis(standardization, axis=0, arr=features)
beta_standardized = leastSquares(standardized_features, labels)
test_cases_standardized = np.apply_along_axis(standardization, axis=0, arr=test_cases)
test_intercept_standardized = np.c_[np.ones(test_cases_standardized.shape[0]), test_cases_standardized]
predicted_temperature_standardized = np.dot(test_intercept_standardized, beta_standardized)
print("Standardized Coefficients (Beta):", beta_standardized)
print("Predicted Temperature (Standardized):", predicted_temperature_standardized)
draw(test_intercept_standardized, beta_standardized, labels)
