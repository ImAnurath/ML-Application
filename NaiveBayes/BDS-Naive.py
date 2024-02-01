import numpy as np
import pandas as pd
import os

def generate_test_case(dataset, num_test_cases):
    num_columns = dataset.shape[1] - 1
    random_indices = np.random.choice(len(dataset), num_test_cases, replace=True)
    random_test_cases = []
    for index in random_indices:
        test_case = dataset.iloc[index, :num_columns].values
        random_test_cases.append(test_case)

    return np.array(random_test_cases)

def normalization(x): # Probability normalization
    exp_x = np.exp(x - np.max(x))
    return exp_x / exp_x.sum()

def bayesClassifier(test_case, features, labels):
    label_counts, results, counts = {}, {}, {}
    label_names, l_counts = np.unique(labels, return_counts=True)

    for index, label_name in enumerate(label_names):
        label_counts[label_name] = l_counts[index]
        results[label_name] = 0  # Initialize as 0 for log-probabilities
        counts[label_name] = 0

    label_probabilities = {label: count / len(labels) for label, count in label_counts.items()}

    epsilon = 1e-9
    for column in range(len(features[0])):
        if isinstance(test_case[column], (int, float)):
            for label in label_counts:  # Numerics
                
                # Gaussian probability calculation for numeric values since I can not turn them into binary or string
                mean = np.mean(features[labels == label, column]) # Numerics mean
                std = np.std(features[labels == label, column])   # Numerics standart deviation
                # Probability density for numerical values (PDF)
                # Ref: https://en.wikipedia.org/wiki/Probability_density_function
                likelihood = (1 / (np.sqrt(2 * np.pi) * std)) * np.exp(-(test_case[column] - mean)**2 / (2 * std**2))
                
                results[label] += np.log(likelihood + epsilon)
        else:  # Binary
            for label in label_counts:
                counts[label] = np.sum((features[:, column] == test_case[column]) & (labels == label))
                results[label] += np.log((counts[label] + epsilon) / (label_counts[label] + epsilon))

    for label in label_probabilities:
        results[label] += np.log(label_probabilities[label])

    results = {label: np.exp(value) for label, value in results.items()}
    normalized_results = normalization(list(results.values())) # Result normalization to avoid very small numbers

    return {label: probability for label, probability in zip(results.keys(), normalized_results)}

script_path = os.path.dirname(os.path.realpath(__file__))
# Data Manipulation
data_name = "milknew.csv"
data_path = os.path.join(script_path, data_name)
data = pd.read_csv(data_path)
label_column = data["Grade"]
labels = label_column.values
features = data.iloc[:, :-1].values

test_cases1 = np.array([
                        #Cases that are taken out of the dataset with their original labels
                         [8.6, 55,  0, 1, 1, 1, 255], #low
                         [3.0, 40,  1, 1, 1, 1, 255], #low
                         [6.7, 45,  1, 1, 0, 0, 247], #medium
                         [6.5, 40,  1, 0, 0, 0, 250], #medium
                         [6.8, 43,  1, 0, 1, 0, 250], #high
                         [6.7, 38,  1, 0, 1, 0, 255], #high
                       ])
num_test_cases = 6
random_test_cases = generate_test_case(data, num_test_cases)

for index, test_case in enumerate(test_cases1):
    result = bayesClassifier(test_case, features, labels)
    
    values = np.array(list(result.values()))
    max_index = np.argmax(values)
    max_key = list(result.keys())[max_index]
    
    print(f"Test Case {index+1}: Highest probability label {max_key}, with {round(result[max_key],3)}")
print("============================")
for index, r_test_case in enumerate(random_test_cases):
    result = bayesClassifier(r_test_case, features, labels)
    
    values = np.array(list(result.values()))
    max_index = np.argmax(values)
    max_key = list(result.keys())[max_index]
    
    print(f"Random Test Case {index+1}: Highest probability label {max_key}, with {round(result[max_key],3)}")