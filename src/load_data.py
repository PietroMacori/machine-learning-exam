# LOAD AND SPLIT DATA
#
# Functions to load the dataset some information on the data (such as attribute names, number of classes...)


import numpy as np

# Information about the dataset
attributes_names = ["Fixed acidity", "Volatile acidity", "Citric acidity", "Residual sugar", "Chlorides",
                    "Free sulfur dioxide", "Total sulfur dioxide", "Density", "pH", "Sulphates", "Alcohol"]
n_attr = len(attributes_names)
class_names = ["Low quality", "High quality"]
n_class = len(class_names)


# Load dataset from textual file in a specified format (one sample per line,
# features are comma-separated)
def load(fileName):
    class_labels_list = []
    list_of_vectors = []

    with open(fileName) as f:
        for line in f:
            try:
                current_vector = np.array(line.split(",")[0:n_attr], dtype=np.float).reshape((n_attr, 1))
                list_of_vectors.append(current_vector)
                class_labels_list.append(int(line.split(",")[n_attr]))
            except:
                pass

    data_matrix = np.array(list_of_vectors).reshape((len(list_of_vectors), n_attr)).T
    class_labels = np.array(class_labels_list)

    return data_matrix, class_labels

