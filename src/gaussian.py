# GAUSSIAN MODELS
#
# This file contains the functions needed to train and test 4 Gaussian models: MVG, Naive Bayes,
# tied MVG and tied Naive Bayes.
# Their performances are evaluated using K_fold (5-fold used) on different type of data: RAW, Gaussianized,
# and Z-Normalized. Each with and without applying the Principal Component Analysis.
# The results (minDCF) of every case has been stored on a txt file.


import numpy as np
from utility import vcol
from load_data import load, n_class
from DCF import compute_min_DCF
from pca import compute_pca
from preprocessing import Z_score


# Compute the log-likelihood ratio given array (2 x N) of log-likelihood
def compute_llr(ll):
    if ll.shape[0] != 2:
        return 0
    # Ratio becomes difference in log
    return ll[1, :] - ll[0, :]


# Compute log-likelihood of an array of samples
def compute_log_likelihood(X, mu, C):
    M = X.shape[0]
    P = np.linalg.inv(C)
    ll = - 0.5 * M * np.log(2 * np.pi)
    ll += -0.5 * np.linalg.slogdet(C)[1]
    lambda_x = np.dot(P, (X - mu))
    points = list()
    for j in range(lambda_x.shape[1]):
        tot = 0
        for i in range(lambda_x.shape[0]):
            tot += (X - mu)[i][j] * lambda_x[i][j]
        points.append(tot)

    ll += -0.5 * np.array(points)

    return ll.ravel()


# Multivariate Gaussian Distribution Model, given training samples and labels compute the
# parameters of the Gaussian and compute log-likelihood ratio of the test samples
def compute_MVG(DTR, LTR, DTE):
    # Mean and covariance of each class and store it in dict
    h = dict()
    for j in range(n_class):
        DI = DTR[:, LTR == j]
        mu_c = vcol(DI.mean(1))
        DIC = DI - mu_c
        CN = 1 / DIC.shape[1] * np.dot(DIC, DIC.T)
        # Store in dict
        h[j] = (mu_c, CN)

    # For each class compute likelihood of test samples
    log_S = np.zeros((n_class, DTE.shape[1]))
    for i in range(n_class):
        # Retrieve class parameters
        mu_c, CN = h[i]
        log_S[i, :] = compute_log_likelihood(DTE, mu_c, CN)

    # Compute log-likelihood ratio
    llr = compute_llr(log_S)

    return llr


# Model of Gaussian with tied covariance assumption (same covariance matrix for all classes)
def compute_Tied(DTR, LTR, DTE):
    h = dict()
    tied_C = 0
    for j in range(n_class):
        DI = DTR[:, LTR == j]
        mu_c = vcol(DI.mean(1))
        DIC = DI - mu_c
        CN = 1 / DIC.shape[1] * np.dot(DIC, DIC.T)
        nc = DIC.shape[1]
        # Compute tied cov matrix = weighted sum of all cov matrix
        tied_C += nc * CN
        # Store only mean value since cov is in common
        h[j] = mu_c

    # Class conditional log probabilities
    log_SJoin = np.zeros((n_class, DTE.shape[1]))
    for i in range(n_class):
        mu_c = h[i]
        log_SJoin[i, :] = compute_log_likelihood(DTE, mu_c, tied_C)

    # Compute log-likelihood ratio
    llr = compute_llr(log_SJoin)

    return llr


# Model of Gaussian with Naive Bayes assumption (diagonal covariance matrix)
# Supposed high independence among components
def compute_Naive_Bayes(DTR, LTR, DTE):
    h = dict()
    for j in range(n_class):
        DI = DTR[:, LTR == j]
        mu_c = vcol(DI.mean(1))
        DIC = DI - mu_c
        CN = 1 / DIC.shape[1] * np.dot(DIC, DIC.T)
        # Remove non-diagonal elements
        CN_diagonal = CN * np.eye(CN.shape[0])
        h[j] = (mu_c, CN_diagonal)

    # Class conditional log probabilities
    log_SJoin = np.zeros((n_class, DTE.shape[1]))
    for i in range(n_class):
        mu_c, CN = h[i]
        log_SJoin[i, :] = compute_log_likelihood(DTE, mu_c, CN)

    # Compute log-likelihood ratio
    llr = compute_llr(log_SJoin)

    return llr


# Model assuming both tied and Naive Bayes assumptions
def compute_Naive_Tied(DTR, LTR, DTE):
    h = dict()
    tied_C = 0
    for j in range(n_class):
        DI = DTR[:, LTR == j]
        mu_c = vcol(DI.mean(1))
        DIC = DI - mu_c
        CN = 1 / DIC.shape[1] * np.dot(DIC, DIC.T)
        nc = DIC.shape[1]
        # Apply tied -> weighted sum of cov matrix
        tied_C += nc * CN
        h[j] = mu_c

    # Apply Bayes -> only diagonal considered
    tied_C *= np.eye(tied_C.shape[0])

    # Class conditional log probabilities
    log_SJoin = np.zeros((n_class, DTE.shape[1]))
    for i in range(n_class):
        mu_c = h[i]
        log_SJoin[i, :] = compute_log_likelihood(DTE, mu_c, tied_C)

    # Compute log-likelihood ratio
    llr = compute_llr(log_SJoin)
    return llr


# K fold function to train and evaluate our models in a precise way
# The number of folds can be imposed by the K argument
def k_fold(Data, Labels, K, gaussian_model, pi, Cfp, Cfn, seed=0):
    # Number of samples of a single fold
    fold_dim = Data.shape[1] // K
    start_idx = 0

    # Shuffle data
    np.random.seed(seed)
    idx = np.random.permutation(Data.shape[1])

    # Define array as long as the training data since with K fold all samples are used to test the model
    llr = np.zeros((Data.shape[1],))

    for i in range(K):
        # Define the end of the test fold
        # Manage the case last fold is smaller than the others
        if start_idx + fold_dim > Data.shape[1]:
            end_idx = Data.shape[1]
        else:
            end_idx = start_idx + fold_dim

        # Define index of train as everything outside (start_index, end_idx)
        idxTrain = np.concatenate((idx[0:start_idx], idx[end_idx:]))
        idxTest = idx[start_idx:end_idx]

        # Define train samples and labels
        DTR = Data[:, idxTrain]
        LTR = Labels[idxTrain]

        # Define test samples
        DTE = Data[:, idxTest]

        # Train the classifier and compute llr on the current test partition
        llr[idxTest] = gaussian_model(DTR, LTR, DTE)
        # Update test fold position
        start_idx += fold_dim

    # Evaluate results after k-fold
    minDCF, _ = compute_min_DCF(llr, Labels, pi, Cfn, Cfp)

    return minDCF


if __name__ == "__main__":
    # Load raw data
    D, L = load("../Data/Train.txt")
    D = Z_score(D)
    # Load gaussianized data
    D_Gauss = np.load("../Data/gaus_data/gaus_train.npy")
    # Compute PCA on gaussianized data
    DG10 = compute_pca(D_Gauss, 10)
    DG9 = compute_pca(D_Gauss, 9)

    # Define parameters
    k = 5
    pi = 0.5
    Cfn = 1
    Cfp = 1
    fileName = "../Results/gaussian_results.txt"

    with open(fileName, "w") as f:
        # Gaussianized - no PCA
        f.write("Gaussianized features - no PCA\n")
        DCF_compute_MVG = round(k_fold(D_Gauss, L, k, compute_MVG, pi, Cfp, Cfn), 3)
        DCF_naive_Bayes = round(k_fold(D_Gauss, L, k, compute_Naive_Bayes, pi, Cfp, Cfn), 3)
        DCF_tied_compute_MVG = round(k_fold(D_Gauss, L, k, compute_Tied, pi, Cfp, Cfn), 3)
        DCF_tied_naive_Bayes = round(k_fold(D_Gauss, L, k, compute_Naive_Tied, pi, Cfp, Cfn), 3)

        f.write("MVG: " + str(DCF_compute_MVG) + "\t\tNaive Bayes: " + str(DCF_naive_Bayes) +
                "\t\tTied: " + str(DCF_tied_compute_MVG) + "\t\tTied Naive Bayes: " + str(DCF_tied_naive_Bayes))

        # Gaussianized - PCA = 10
        f.write("\n\nGaussianized features - PCA = 10\n")
        DCF_compute_MVG = round(k_fold(DG10, L, k, compute_MVG, pi, Cfp, Cfn), 3)
        DCF_naive_Bayes = round(k_fold(DG10, L, k, compute_Naive_Bayes, pi, Cfp, Cfn), 3)
        DCF_tied_compute_MVG = round(k_fold(DG10, L, k, compute_Tied, pi, Cfp, Cfn), 3)
        DCF_tied_naive_Bayes = round(k_fold(DG10, L, k, compute_Naive_Tied, pi, Cfp, Cfn), 3)

        f.write("MVG: " + str(DCF_compute_MVG) + "\t\tNaive Bayes: " + str(DCF_naive_Bayes) +
                "\t\tTied: " + str(DCF_tied_compute_MVG) + "\t\tTied Naive Bayes: " + str(DCF_tied_naive_Bayes))

        # Gaussianized - PCA = 9
        f.write("\n\nGaussianized features - PCA = 9\n")
        DCF_compute_MVG = round(k_fold(DG9, L, k, compute_MVG, pi, Cfp, Cfn), 3)
        DCF_naive_Bayes = round(k_fold(DG9, L, k, compute_Naive_Bayes, pi, Cfp, Cfn), 3)
        DCF_tied_compute_MVG = round(k_fold(DG9, L, k, compute_Tied, pi, Cfp, Cfn), 3)
        DCF_tied_naive_Bayes = round(k_fold(DG9, L, k, compute_Naive_Tied, pi, Cfp, Cfn), 3)

        f.write("MVG: " + str(DCF_compute_MVG) + "\t\tNaive Bayes: " + str(DCF_naive_Bayes) +
                "\t\tTied: " + str(DCF_tied_compute_MVG) + "\t\tTied Naive Bayes: " + str(DCF_tied_naive_Bayes))

        # Raw features - no PCA
        f.write("\n\nRaw features - no PCA\n")
        DCF_compute_MVG = round(k_fold(D, L, k, compute_MVG, pi, Cfp, Cfn), 3)
        DCF_naive_Bayes = round(k_fold(D, L, k, compute_Naive_Bayes, pi, Cfp, Cfn), 3)
        DCF_tied_compute_MVG = round(k_fold(D, L, k, compute_Tied, pi, Cfp, Cfn), 3)
        DCF_tied_naive_Bayes = round(k_fold(D, L, k, compute_Naive_Tied, pi, Cfp, Cfn), 3)

        f.write("MVG: " + str(DCF_compute_MVG) + "\t\tNaive Bayes: " + str(DCF_naive_Bayes) +
                "\t\tTied: " + str(DCF_tied_compute_MVG) + "\t\tTied Naive Bayes: " + str(DCF_tied_naive_Bayes))

