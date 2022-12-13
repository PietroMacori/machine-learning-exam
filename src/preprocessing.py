# DATA PREPROCESSING
#
# This file contains the functions needed to perform some pre-processing steps such as Gaussianization
# and Z-normalization of the samples. Moreover, it provides some function needed for the data visualization
# task

import numpy as np
import matplotlib.pyplot as plt
from load_data import load, attributes_names, class_names
from scipy.stats import norm
import seaborn


# Save histograms of the features of dataset
def print_histograms(data, labels, figName):
    for i, attribute in enumerate(attributes_names):
        plt.figure()
        plt.xlabel(attribute)
        plt.title("Histogram of feature: " + attribute)
        for curr_class in range(len(class_names)):
            mask = (labels == curr_class)
            plt.hist(data[i, mask], bins=20, density=True, ec='black', alpha=0.5)
        plt.legend(class_names)
        plt.savefig("../Images/Hist/" + figName + str(i) + ".png")
        print("Saving  " + figName + str(i) + ".png\t DONE")
        plt.close()


# Apply gaussianization, depending on the presence of DTE, compute Train or Test gaussianization
def gaussianize_features(DTR, DTE=None):
    if DTE is None:
        # Gaussianization of TRAIN
        gauss_feat_matrix = np.zeros(DTR.shape)
        for feature_idx in range(DTR.shape[0]):
            for sample_idx in range(DTR.shape[1]):
                r = (sum(DTR[feature_idx, sample_idx] < DTR[feature_idx, :]) + 1) / (DTR.shape[1] + 2)
                gauss_feat_matrix[feature_idx, sample_idx] = norm.ppf(r)
        return gauss_feat_matrix
    else:
        # Gaussianization of TEST using train data
        gauss_feat_matrix = np.zeros(DTE.shape)
        for feat in range(DTE.shape[0]):
            for sample in range(DTE.shape[1]):
                gauss_feat_matrix[feat, sample] = norm.ppf(
                    (sum(DTE[feat, sample] < DTR[feat, :]) + 1) / (DTR.shape[1] + 2))
    return gauss_feat_matrix


# Save the heatmap of feature correlation
def print_heatmap(D, figName):
    plt.figure()
    seaborn.heatmap(np.corrcoef(D))
    plt.savefig("../Images/Heat/" + figName + ".png")
    plt.close()


# Apply Z-score normalization to train or test data using mean and variance of training dataset
def Z_score(DTR, DTE=None):
    if DTE is None:
        return (DTR - DTR.mean(1).reshape((DTR.shape[0], 1))) / (np.var(DTR, axis=1).reshape((DTR.shape[0], 1)) ** 0.5)
    else:
        return (DTE - DTR.mean(1).reshape((DTR.shape[0], 1))) / (np.var(DTR, axis=1).reshape((DTR.shape[0], 1)) ** 0.5)


if __name__ == "__main__":
    # Load train and test data
    DTR, LTR = load("../Data/Train.txt")
    DTE, LTE = load("../Data/Test.txt")

    print("# Bad quality:", sum(LTR == 0))
    print("# Good quality:", sum(LTR == 1), "\n")

    # Distributions of features
    print_histograms(DTR, LTR, "Raw_Feat_Hist_")

    # Gaussianize features and save them
    gauss_feat = gaussianize_features(DTR)
    np.save("../Data/gaus_data/gaus_train.npy", gauss_feat)
    print_histograms(gauss_feat, LTR, "Gauss_Feat_Hist_")

    # Print heatmap of correlation of whole dataset and of each class for RAW data
    print_heatmap(DTR, "Raw_Heat_Dataset")
    print_heatmap(DTR[:, LTR == 0], "Raw_Heat_Bad")
    print_heatmap(DTR[:, LTR == 1], "Raw_Heat_Good")

    # Print heatmap of correlation of whole dataset and of each class for gaus_data data
    print_heatmap(gauss_feat, "Gauss_Heat_Dataset")
    print_heatmap(gauss_feat[:, LTR == 0], "Gauss_Heat_Bad")
    print_heatmap(gauss_feat[:, LTR == 1], "Gauss_Heat_Good")

    # Test data must be gaussianized using training
    DTE_gauss = gaussianize_features(DTR, DTE)
    np.save("../Data/gaus_data/gaus_test.npy", DTE_gauss)

    # Distributions of gaussianized features
    print_histograms(DTE_gauss, LTE, "GaussFeatHist")
