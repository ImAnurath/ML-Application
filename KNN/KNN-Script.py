import numpy as  np
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

def euclidean_distance(train_point, test_point):
    train_point = np.asarray(train_point, dtype=np.float64)
    test_point = np.asarray(test_point, dtype=np.float64)
    return np.sqrt(np.sum((train_point - test_point) ** 2))

def knn_classifier(train_features, train_labels, test_point, k):
    distances = np.zeros(len(train_labels))
    for i, train_feature_point in enumerate(train_features):
        distances[i] = euclidean_distance(test_point, train_feature_point)

    nearest_indices = np.argsort(distances)[:k]
    nearest_labels = train_labels[nearest_indices]
    unique_labels, counts = np.unique(nearest_labels, return_counts=True)
    max_count = np.max(counts)

    if np.sum(counts == max_count) > 1:
        return 'UNIDENTIFIED'
    else:
        predicted_label = unique_labels[np.argmax(counts)]
        return predicted_label

script_path = os.path.dirname(os.path.realpath(__file__))
# Data Manipulation
data_name = "IRIS.csv"
data_path  = os.path.join(script_path, data_name)
data  = pd.read_csv(data_path)
labels = data["species"].values
features = data.iloc[:, :-1].values

k_value = 3

test_cases0 = np.array([ # Test cases that are INSIDE the dataset

                        [5,   3,   1.6, 0.2],  # Setosa     - OG Index: 27
                        [4.8, 3.4, 1.6, 0.2],  # Setosa     - OG Index: 13
                        [6.2, 2.9, 4.3, 1.3],  # Versicolor - OG Index: 99
                        [5.8, 2.7, 4.1, 1],    # Versicolor - OG Index: 69
                        [7.2, 3,   5.8, 1.6],  # Virginica  - OG Index: 131
                        [6.4, 3.1, 5.5, 1.8],  # Virginica  - OG Index: 139
                       ])

test_cases1 = np.array([ # Test cases that are TAKEN OUT of the dataset

                        [5,   3.3, 1.4, 0.2],  # Setosa     - OG Index: 51
                        [5.7, 2.8, 4.1, 1.3],  # Versicolor - OG Index: 100
                        [5.9, 3,   5.1, 1.8],  # Virginica: - OG Index: 151 
                       ])
num_test_cases = 6
random_test_cases = generate_test_case(data, num_test_cases)

for index,test_case in enumerate(test_cases0):
    predicted_label = knn_classifier(features, labels, test_case, k_value)
    print(f"Result of cases that are inside: Case {index+1}: {predicted_label}")
print("=====================================")
for index,test_case in enumerate(test_cases1):
    predicted_label = knn_classifier(features, labels, test_case, k_value)
    print(f"Result of cases that are taken out: Case {index+1}: {predicted_label}")
print("=====================================")
for index,test_case in enumerate(random_test_cases):
    predicted_label = knn_classifier(features, labels, test_case, k_value)
    print(f"Result of cases that are randomly generated: Case {random_test_cases[index]}: {predicted_label}")