# THRESHOLD AND MODEL COMBINATION
#
# This file contains the functions needed to compute the actual DCF using the optimal threshold and
# estimating the threshold for the specific application. Moreover, the fusion of the single models is
# performed in this file. The actual DCF computation and the fusion are performed on:
# - Quadratic Linear Regression, lambda = 0, Z-normalized features, no PCA
# - RBF SVM with C = 10 and log(gamma) = -2, rebalanced and Z-normalized features
# - Gaussian Mixture Model, Z-normalized features, 8 components


import numpy as np
from load_data import load
from DCF import compute_act_DCF, compute_min_DCF, compute_binary_confusion, norm_bayes_risk
import svm as SVM
from preprocessing import Z_score
import logistic_regression as LR
import gmm as GMM
from utility import vrow


# Perform cross validation to fuse 3 models (scores combined using linear logistic regression)
def k_fold_fusion_3_models(llr_1, llr_2, llr_3, L, K, pi, Cfp, Cfn, pi_T, l, seed=0):
    # Number of samples of a single fold
    fold_dim = llr_1.shape[1] // K
    start_idx = 0

    # Shuffle data
    np.random.seed(seed)
    idx = np.random.permutation(llr_1.shape[1])

    # Define array as long as the training data since with K fold all samples are used to test the model
    llr = np.zeros((llr_1.shape[1],))

    for i in range(K):
        # Define the end of the test fold
        # Manage the case last fold is smaller than the others
        if start_idx + fold_dim > llr_1.shape[1]:
            end_idx = llr_1.shape[1]
        else:
            end_idx = start_idx + fold_dim

        # Define index of train as everything outside (start_index, end_idx)
        idxTrain = np.concatenate((idx[0:start_idx], idx[end_idx:]))
        idxTest = idx[start_idx:end_idx]

        # Define training samples combining the scores of the three models
        DTR = np.zeros([3, idxTrain.shape[0]])
        DTR[0, :] = llr_1[:, idxTrain].reshape([llr_1[:, idxTrain].shape[1], ])
        DTR[1, :] = llr_2[:, idxTrain].reshape([llr_2[:, idxTrain].shape[1], ])
        DTR[2, :] = llr_3[:, idxTrain].reshape([llr_3[:, idxTrain].shape[1], ])
        DTE = np.zeros([3, idxTest.shape[0]])
        DTE[0, :] = llr_1[:, idxTest].reshape([llr_1[:, idxTest].shape[1], ])
        DTE[1, :] = llr_2[:, idxTest].reshape([llr_2[:, idxTest].shape[1], ])
        DTE[2, :] = llr_3[:, idxTest].reshape([llr_3[:, idxTest].shape[1], ])

        LTR = L[idxTrain]

        # Train a logistic regression model
        llr[idxTest] = LR.linear_logistic(DTR, LTR, DTE, l, pi_T)

        start_idx += fold_dim

    # Calculate min and act DCF for the fusion
    minDCF, _ = compute_min_DCF(llr, L, pi, Cfn, Cfp)
    actDCF = compute_act_DCF(llr, L, pi, Cfn, Cfp)

    return minDCF, actDCF


# Perform cross validation to fuse 2 models (scores combined using linear logistic regression)
def k_fold_fusion_2_models(llr_1, llr_2, Labels, K, pi, Cfp, Cfn, pi_T, l, seed=0):
    # Number of samples of a single fold
    fold_dim = llr_1.shape[1] // K
    start_idx = 0

    # Shuffle data
    np.random.seed(seed)
    idx = np.random.permutation(llr_1.shape[1])

    # Define array as long as the training data since with K fold all samples are used to test the model
    llr = np.zeros((llr_1.shape[1],))

    for i in range(K):
        # Define the end of the test fold
        # Manage the case last fold is smaller than the others
        if start_idx + fold_dim > llr_1.shape[1]:
            end_idx = llr_1.shape[1]
        else:
            end_idx = start_idx + fold_dim

        # Define index of train as everything outside (start_index, end_idx)
        idxTrain = np.concatenate((idx[0:start_idx], idx[end_idx:]))
        idxTest = idx[start_idx:end_idx]

        # Define training and test sets combining the scores of models
        DTR = np.zeros([2, idxTrain.shape[0]])
        DTE = np.zeros([2, idxTest.shape[0]])
        DTR[0, :] = llr_1[:, idxTrain].reshape([llr_1[:, idxTrain].shape[1], ])
        DTR[1, :] = llr_2[:, idxTrain].reshape([llr_2[:, idxTrain].shape[1], ])
        DTE[0, :] = llr_1[:, idxTest].reshape([llr_1[:, idxTest].shape[1], ])
        DTE[1, :] = llr_2[:, idxTest].reshape([llr_2[:, idxTest].shape[1], ])

        LTR = Labels[idxTrain]

        # Train a logistic regression model
        llr[idxTest] = LR.linear_logistic(DTR, LTR, DTE, l, pi_T)

        start_idx += fold_dim

    # Calculate min and act DCF for the fusion
    minDCF, _ = compute_min_DCF(llr, Labels, pi, Cfn, Cfp)
    actDCF = compute_act_DCF(llr, Labels, pi, Cfn, Cfp)

    return minDCF, actDCF


# Perform cross validation to evaluate score calibration
def k_fold_calibration(llr, Labels, K, pi, Cfp, Cfn, seed=0):
    # Number of samples of a single fold
    fold_dim = llr.shape[1] // K
    start_idx = 0

    # Shuffle data
    np.random.seed(seed)
    idx = np.random.permutation(llr.shape[1])

    # Define array as long as the training data since with K fold all samples are used to test the model
    opt_th_decisions = np.zeros((llr.shape[1],))

    for i in range(K):
        # Define the end of the test fold
        # Manage the case last fold is smaller than the others
        if start_idx + fold_dim > llr.shape[1]:
            end_idx = llr.shape[1]
        else:
            end_idx = start_idx + fold_dim

        # Define index of train as everything outside (start_index, end_idx)
        idxTrain = np.concatenate((idx[0:start_idx], idx[end_idx:]))
        idxTest = idx[start_idx:end_idx]

        # Define train samples and labels
        DTR = llr[:, idxTrain]
        LTR = Labels[idxTrain]

        # Define test samples
        DTE = llr[:, idxTest]

        # Compute the threshold related to the minimum DCF for the training set
        _, opt_t = compute_min_DCF(DTR.ravel(), LTR, pi, Cfn, Cfp)
        # Evaluate test set using optimal threshold
        opt_th_decisions[idxTest] = DTE.ravel() > opt_t
        start_idx += fold_dim

    # Calculate act DCF for optimal estimated threshold
    M = compute_binary_confusion(opt_th_decisions, L)
    actDCF_estimated = norm_bayes_risk(M, pi, Cfn, Cfp)

    return actDCF_estimated


# Report the actual DCF using the theoretical optimal threshold and the estimated one
def analyse_scores(llr, Labels, pi, Cfn, Cfp, k):
    # Compute the actual DCF
    actDCF = compute_act_DCF(llr, L, pi, Cfn, Cfp)

    # Estimate DCF
    actDCF_estimated = k_fold_calibration(vrow(llr), Labels, k, pi, Cfp, Cfn)

    return actDCF, actDCF_estimated


# Perform cross validation to evaluate fusion of 2 models and print results
def compute_DCF_2_models(D1, D2, L, k, pi, Cfp, Cfn, pi_T):
    # Choose the best value for lambda for logistic regression (try different ones)
    best_minDCF = 1
    best_actDCF = 1
    best_lambda = 0
    for l in [0, 1e-6, 1e-3, 0.1, 1]:
        minDCF, actDCF = k_fold_fusion_2_models(D1.reshape([1, D1.shape[0]]), D2.reshape([1, D2.shape[0]]), L, k, pi,
                                                Cfp, Cfn, pi_T, l)
        if minDCF < best_minDCF:
            best_minDCF = minDCF
            best_actDCF = actDCF
            best_lambda = l

    return best_minDCF, best_actDCF, best_lambda


# Perform cross validation to evaluate fusion of 3 models and print results
def compute_DCF_3_models(D1, D2, D3, L, k, pi, Cfp, Cfn, pi_T):
    # Choose the best value for lambda for logistic regression (try different ones)
    best_minDCF = 1
    best_actDCF = 1
    best_lambda = 0
    for l in [0, 1e-6, 1e-3, 0.1, 1]:
        minDCF, actDCF = k_fold_fusion_3_models(D1.reshape([1, D1.shape[0]]), D2.reshape([1, D2.shape[0]]),
                                                D3.reshape([1, D3.shape[0]]), L, k, pi, Cfp, Cfn, pi_T, l)
        if minDCF < best_minDCF:
            best_minDCF = minDCF
            best_actDCF = actDCF
            best_lambda = l

    return best_minDCF, best_actDCF, best_lambda


if __name__ == "__main__":
    # Load data
    D, L = load("../Data/Train.txt")
    DN = Z_score(D)

    # Define parameters
    k = 5
    pi = 0.5
    Cfp = 1
    Cfn = 1
    pi_T = 0.5

    # Quadratic Linear Regression, lambda = 0, Z-normalized features, no PCA
    _, llrLR = LR.k_fold(DN, L, LR.quadratic_logistic, k, pi, Cfp, Cfn, 0, pi_T)
    np.save("../Data/llrLR.npy", llrLR)
    act, optAct = analyse_scores(llrLR, L, pi, Cfn, Cfp, k)
    print("\n\n------- LR -------\n")
    print("act DCF, optimal threshold: " + str(act))
    print("act DCF, estimated threshold: " + str(optAct))

    # RBF SVM with C = 10 and log(gamma) = -2, rebalanced and Z-normalized features
    _, llrSVM = SVM.k_fold(DN, L, k, pi, Cfp, Cfn, 10, pi_T, 1, balance=True, kernel=SVM.kernel_rbf,
                           args_kernel=[np.exp(-2), 1 ** 0.5])
    np.save("../Data/llrSVM.npy", llrSVM)
    act, optAct = analyse_scores(llrSVM, L, pi, Cfn, Cfp, k)
    print("\n\n------- SVM -------\n")
    print("act DCF, optimal threshold: " + str(act))
    print("act DCF, estimated threshold: " + str(optAct))

    # Gaussian Mixture Model, Z-normalized features, 8 components
    _, llrGMM = GMM.k_fold(DN, L, k, pi, Cfp, Cfn, False, False, 8, seed=0)
    np.save("../Data/llrGMM.npy", llrGMM)
    act, optAct = analyse_scores(llrGMM, L, pi, Cfn, Cfp, k)
    print("\n\n------- GMM -------\n")
    print("act DCF, optimal threshold: " + str(act))
    print("act DCF, estimated threshold: " + str(optAct))

    # Compute min and actual DCF for combined models
    minDCF, actDCF, l = compute_DCF_2_models(llrSVM, llrLR, L, k, pi, Cfp, Cfn, pi_T)
    print("\n\n----- SVM + LR -----")
    print("min DCF: " + str(minDCF))
    print("act DCF: " + str(actDCF))
    print("best lambda: " + str(l))

    minDCF, actDCF, l = compute_DCF_2_models(llrSVM, llrGMM, L, k, pi, Cfp, Cfn, pi_T)
    print("\n\n----- SVM + GMM -----")
    print("min DCF: " + str(minDCF))
    print("act DCF: " + str(actDCF))
    print("best lambda: " + str(l))

    minDCF, actDCF, l = compute_DCF_3_models(llrSVM, llrLR, llrGMM, L, k, pi, Cfp, Cfn, pi_T)
    print("\n\n----- SVM + LR + GMM -----")
    print("min DCF: " + str(minDCF))
    print("act DCF: " + str(actDCF))
    print("best lambda: " + str(l))
