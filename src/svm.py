# SUPPORT VECTOR MACHINE
#
# This file contains the functions needed to train and test different flavours of SVM classifiers, specifically
# a linear SVM and two kernel-based SVM (polynomial and gaussian radial basis). For sake of simplicity and
# re-usability a unique function (train_SVM) has been implemented for all three cases. The models are tested
# using differently pre-processed data, exploiting the k-fold cross validation technique to have a better insight
# on the performances of the models. The k-fold allowed to cross-validate different parameters used in the SVMs to
# understand which one performed better. The minimum DCF function is used to compare the classifiers performances.

import numpy as np
import scipy
from utility import vrow, vcol
from load_data import load
from preprocessing import Z_score
from DCF import compute_min_DCF
import matplotlib.pyplot as plt


# Definition of the polynomial kernel -> (x1 * x2 + c) ^ d + bias
# args is a list composed of 3 elements containing c, d and bias
def kernel_polynomial(X1, X2, args):
    # args is a list containing three entries: c, d and bias
    c = args[0]
    d = args[1]
    bias = args[2]
    return (np.dot(X1.T, X2) + c) ** d + bias


# Definition of the Gaussian Radial Basis kernel -> e^(-gamma * ||x1 - x2||^2) + bias
# args is a list composed of 2 elements containing gamma and bias
def kernel_rbf(X1, X2, args):
    # args is a list containing two entries: gamma and bias
    gamma = args[0]
    bias = args[1]
    return np.exp(- gamma * np.linalg.norm(X1 - X2) ** 2) + bias


# Function to compute the H matrix for both linear and kernel case
def compute_H(X, Z, kernel_function=None, args_kernel=None):
    H = np.zeros((X.shape[1], X.shape[1]))

    if kernel_function is None or args_kernel is None:
        # Linear case
        H = np.dot(X.T, X)
        H = vcol(Z) * vrow(Z) * H
    else:
        # Kernel case
        for i in range(X.shape[1]):
            for j in range(X.shape[1]):
                H[i][j] = Z[i] * Z[j] * kernel_function(X[:, i], X[:, j], args_kernel)
    return H


# Train the generic SVM. The kernel and args_kernel arguments allow to understand the type of SVM
def train_SVM(DTR, LTR, DTE, C, K, pi_T, balance=False, kernel=None, args_kernel=None):
    if kernel is None or args_kernel is None:
        # Modify training data to avoid additional constrain in dual problem
        x_hat = np.vstack([DTR, K * np.ones((1, DTR.shape[1]))])
    else:
        x_hat = DTR

    # Compute Z mapping Ht -> 1 and Hf -> -1
    Z = 2 * LTR - 1

    # Compute H_hat matrix (modified version of H matrix avoiding addition constrain)
    H_hat = compute_H(x_hat, Z, kernel, args_kernel)

    # Dual problem function: compute function and gradient for numerical solver
    def JDual(alpha):
        Ha = np.dot(H_hat, vcol(alpha))
        aHa = np.dot(vrow(alpha), Ha)
        a1 = alpha.sum()
        return -0.5 * aHa.ravel() + a1, - Ha.ravel() + np.ones(alpha.size)

    # Define the L function for dual case as negative of it -> in this way we can minimize instead of maximize
    def LDual(alpha):
        loss, grad = JDual(alpha)
        return -loss, -grad

    # Balance the dataset
    B = np.zeros([DTR.shape[1], 2])
    if balance:
        # Compute pi empirical as samples with label 1 over the dataset dimension
        pi_emp = sum(LTR == 1) / DTR.shape[1]
        # Compute Ct and Cf
        Ct = C * pi_T / pi_emp
        Cf = C * (1 - pi_T) / (1 - pi_emp)
        # Set different boundaries depending on labels
        B[LTR == 1, 1] = Ct
        B[LTR == 0, 1] = Cf
    else:
        B[:, 1] = C

    # Compute the optimal alpha
    alpha_star, _, _ = scipy.optimize.fmin_l_bfgs_b(LDual, np.zeros(DTR.shape[1]),
                                                    approx_grad=False, bounds=B, factr=10000.0)

    # Compute scores
    if kernel is None or args_kernel is None:
        # Linear case
        w_star = np.dot(x_hat, vcol(alpha_star) * vcol(Z))
        # Compute extended data matrix for test dataset
        T_hat = np.concatenate((DTE, K * np.array(np.ones([1, DTE.shape[1]]))))
        # Compute score as dot product between w and extended test matrix
        S = np.dot(w_star.T, T_hat)
    else:
        # Kernel case
        kernel_mat = np.zeros([DTR.shape[1], DTE.shape[1]])
        # Compute matrix storing the dot product of the kernel function
        for index1 in range(DTR.shape[1]):
            for index2 in range(DTE.shape[1]):
                kernel_mat[index1, index2] = kernel(DTR[:, index1], DTE[:, index2], args_kernel)
        # Compute score
        S = np.sum((alpha_star * Z).reshape([DTR.shape[1], 1]) * kernel_mat, axis=0)

    return S.ravel()


# Perform k-fold cross validation on test data for the specified model
def k_fold(Data, Labels, K, pi, Cfp, Cfn, C, pi_T, K_SVM, balance=False, kernel=None, args_kernel=None, seed=0):
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

        # Train the classifier and compute llr on the current partition
        llr[idxTest] = train_SVM(DTR, LTR, DTE, C, K_SVM, pi_T, balance, kernel, args_kernel)

        # Update fold index
        start_idx += fold_dim

    # Compute minimum DCF
    minDCF, _ = compute_min_DCF(llr, Labels, pi, Cfn, Cfp)

    return minDCF, llr


if __name__ == '__main__':
    # Load data
    D, L = load("../Data/Train.txt")
    DN = Z_score(D)
    DG = np.load("../Data/gaus_data/gaus_train.npy")

    # Define parameters
    pi_T = 0.5
    pi = 0.5
    Cfn = 1
    Cfp = 1
    k = 5
    K_SVM = 1
    C_val = [1e-1, 1, 10]

    # Linear
    fileName = "../Results/linear_SVM_results.txt"
    with open(fileName, "w") as f_in:
        f_in.write("C Values:\n1e-2 \t1e-1 \t1    \t10   \t100  \n")
        for balance in [False, True]:
            f_in.write("\n\nBalanced: " + str(balance) + "\n")
            f_in.write("\nRAW\n")
            dcf_raw = list()
            for C in C_val:
                minDCF, _ = k_fold(D, L, k, pi, Cfp, Cfn, C, pi_T, K_SVM, balance=balance)
                dcf_raw.append(minDCF)
                f_in.write(str(round(minDCF, 3)) + "\t")

            f_in.write("\nZ-normalized - no PCA\n")
            dcf_z = list()
            for C in C_val:
                minDCF, _ = k_fold(DN, L, k, pi, Cfp, Cfn, C, pi_T, K_SVM, balance=balance)
                dcf_z.append(minDCF)
                f_in.write(str(round(minDCF, 3)) + "\t")

            f_in.write("\nGaussianized\n")
            dcf_gaus = list()
            for C in C_val:
                minDCF, _ = k_fold(DG, L, k, pi, Cfp, Cfn, C, pi_T, K_SVM, balance=balance)
                dcf_gaus.append(minDCF)
                f_in.write(str(round(minDCF, 3)) + "\t")
            if balance:
                imgName = "SVM_linear_balanced.png"
            else:
                imgName = "SVM_linear_unbalanced.png"

            plt.figure()
            plt.plot(C_val, dcf_raw, marker='o', linestyle='dashed', color="red")
            plt.plot(C_val, dcf_z, marker='o', linestyle='dashed', color="blue")
            plt.plot(C_val, dcf_gaus, marker='o', linestyle='dashed', color="green")
            plt.xscale("log")
            plt.xlabel("C")
            plt.ylabel("min DCF")
            plt.legend(["Raw", "Z-normalized", "Gaussianized"])
            plt.savefig("../Images/" + imgName)
    print("Linear SVM \t\t DONE")

    # Polynomial Kernel
    fileName = "../Results/polynomial_SVM_results.txt"
    with open(fileName, "w") as f_in:
        f_in.write("C Values:\n1e-2 \t1e-1 \t1    \t10   \t100  \n")
        for balance in [False, True]:
            f_in.write("\n\nBalanced: " + str(balance) + "\n")
            f_in.write("\nRAW\n")
            dcf_raw = list()
            for C in C_val:
                minDCF, _ = k_fold(D, L, k, pi, Cfp, Cfn, C, pi_T, K_SVM, balance=balance,
                                   kernel=kernel_polynomial, args_kernel=[1, 2, K_SVM ** 0.5])
                dcf_raw.append(minDCF)
                f_in.write(str(round(minDCF, 3)) + "\t")

            f_in.write("\nZ-normalized - no PCA\n")
            dcf_z = list()
            for C in C_val:
                minDCF, _ = k_fold(DN, L, k, pi, Cfp, Cfn, C, pi_T, K_SVM, balance=balance,
                                   kernel=kernel_polynomial, args_kernel=[1, 2, K_SVM ** 0.5])
                dcf_z.append(minDCF)
                f_in.write(str(round(minDCF, 3)) + "\t")

            f_in.write("\nGaussianized\n")
            dcf_gaus = list()
            for C in C_val:
                minDCF, _ = k_fold(DG, L, k, pi, Cfp, Cfn, C, pi_T, K_SVM, balance=balance,
                                   kernel=kernel_polynomial, args_kernel=[1, 2, K_SVM ** 0.5])
                dcf_gaus.append(minDCF)
                f_in.write(str(round(minDCF, 3)) + "\t")

            if balance:
                imgName = "SVM_polynomial_balanced.png"
            else:
                imgName = "SVM_polynomial_unbalanced.png"

            plt.figure()
            plt.plot(C_val, dcf_gaus, marker='o', linestyle='dashed', color="red")

            plt.xscale("log")
            plt.xlabel("C")
            plt.ylabel("min DCF")
            plt.savefig("../Images/" + imgName)
    print("Quadratic SVM \t\t DONE")

    # RBF Kernel
    fileName = "../Results/RBG_SVM_results.txt"
    gamma = [np.exp(-1), np.exp(-2)]

    with open(fileName, "w") as f_in:
        f_in.write("C Values:\t1e-1 \t1    \t10   \n")
        for balance in [False, True]:
            f_in.write("\n\nBalanced: " + str(balance) + "\n")
            complete_z = list()
            complete_gaus = list()
            for g in gamma:
                f_in.write("\n\nGamma: " + str(g) + "\n")
                f_in.write("\nZ-normalized - no PCA\n")
                dcf_z = list()
                for C in C_val:
                    minDCF, _ = k_fold(DN, L, k, pi, Cfp, Cfn, C, pi_T, K_SVM,
                                       balance=balance,
                                       kernel=kernel_rbf, args_kernel=[g, K_SVM ** 0.5])
                    dcf_z.append(minDCF)
                    f_in.write(str(round(minDCF, 3)) + "\t")
                complete_z.append(dcf_z)

            if balance:
                img = "SVM_RBF_balance.png"
            else:
                img = "SVM_RBF_unbalance.png"

            plt.figure()
            plt.plot(C_val, complete_z[0], marker='o', linestyle='dashed', color="red")
            plt.plot(C_val, complete_z[1], marker='o', linestyle='dashed', color="blue")

            plt.xscale("log")
            plt.xlabel("C")
            plt.ylabel("min DCF")
            plt.legend([r"$log \gamma = -1$",
                        r"$log \gamma = -2$"])
            plt.savefig("../Images/" + img)
    print("RBF SVM \t\t DONE")

