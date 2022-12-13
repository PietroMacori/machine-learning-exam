# MODELS EVALUATION
#
# This file contains the functions needed to evaluate the models implemented for this project using the test data.
# The minDCF is computed for all models and the evaluation of the score calibration is done. Moreover, the ROC
# and Bayes Error plots are done for the best models.

import numpy as np
from load_data import load
from gaussian import compute_MVG, compute_Tied, compute_Naive_Tied, compute_Naive_Bayes
from logistic_regression import linear_logistic, quadratic_logistic
from preprocessing import Z_score
from DCF import compute_min_DCF, compute_act_DCF, compute_binary_confusion, norm_bayes_risk, ROC_components, \
    Bayes_plot_components
from pca import compute_pca
from svm import train_SVM, kernel_rbf, kernel_polynomial
from gmm import GMM_classifier
import matplotlib.pyplot as plt


# Evaluate score calibration by training a logistic regression model on training data and evaluating it on test data
def evaluate_score_calibration(llrTrain, llrTest, LTR, LTE, pi, Cfn, Cfp):
    # Select optimal threshold on training llr and evaluate results on test llr
    _, opt_t = compute_min_DCF(llrTrain, LTR, pi, Cfn, Cfp)
    # Define matrix for predicted labels
    PredictedLabels = np.zeros(LTE.shape)
    PredictedLabels[llrTest > opt_t] = 1
    M = compute_binary_confusion(PredictedLabels, LTE)
    # Compute estimated actual DCF
    actDCF_estim = norm_bayes_risk(M, pi, Cfn, Cfp)

    return actDCF_estim


# Evaluate combination of 2 models by training a logistic regression model on training data and evaluating it on test
# data
def evaluate_2_models_combination(llrTrain1, llrTrain2, llrTest1, llrTest2, LTR, LTE, l):
    # Define DTR and DTE as combinations of llr from different models
    DTR = np.zeros([2, llrTrain1.shape[0]])
    DTR[0, :] = llrTrain1.reshape([llrTrain1.shape[0], ])
    DTR[1, :] = llrTrain2.reshape([llrTrain2.shape[0], ])
    DTE = np.zeros([2, llrTest1.shape[0]])
    DTE[0, :] = llrTest1.reshape([llrTest1.shape[0], ])
    DTE[1, :] = llrTest2.reshape([llrTest2.shape[0], ])

    # Apply linear regression
    s = linear_logistic(DTR, LTR, DTE, l, 0.5)
    minDCF = compute_min_DCF(s, LTE, pi, Cfn, Cfp)
    actDCF = compute_act_DCF(s, LTE, pi, Cfn, Cfp)

    return minDCF, actDCF, s


# Evaluate fusion of 3 models by training a logistic regression model on training data
# and evaluating it on test data
def evaluate_3_models_combination(llrTrain1, llrTrain2, llrTrain3, llrTest1, llrTest2, llrTest3, LTR, LTE, l):
    # Define DTR and DTE as combinations of llr from different models
    DTR = np.zeros([3, llrTrain1.shape[0]])
    DTR[0, :] = llrTrain1.reshape([llrTrain1.shape[0], ])
    DTR[1, :] = llrTrain2.reshape([llrTrain2.shape[0], ])
    DTR[2, :] = llrTrain3.reshape([llrTrain3.shape[0], ])
    DTE = np.zeros([3, llrTest1.shape[0]])
    DTE[0, :] = llrTest1.reshape([llrTest1.shape[0], ])
    DTE[1, :] = llrTest2.reshape([llrTest2.shape[0], ])
    DTE[2, :] = llrTest3.reshape([llrTest3.shape[0], ])

    # Apply linear regression
    s = linear_logistic(DTR, LTR, DTE, l, 0.5)
    minDCF = compute_min_DCF(s, LTE, pi, Cfn, Cfp)
    actDCF = compute_act_DCF(s, LTE, pi, Cfn, Cfp)

    return minDCF, actDCF, s


def plot_ROC_curve(FPR1, TPR1, FPR2, TPR2, FPR3, TPR3, l1, l2, l3, figName):
    plt.figure()
    plt.plot(FPR1, TPR1, 'b')
    plt.plot(FPR2, TPR2, 'r')
    plt.plot(FPR3, TPR3, 'g')
    plt.grid(True)
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.legend([l1, l2, l3])
    plt.savefig("../Images/" + figName + ".png")


def plot_Bayes_error(D1, D2, D3, D4, l1, l2, l3, l4, figName, D5=None, D6=None, l5=None, l6=None):
    effPriorLogOdds = np.linspace(-3, 3, 21)

    plt.figure()
    plt.plot(effPriorLogOdds, D1, label=l1, color='red')
    plt.plot(effPriorLogOdds, D2, label=l2, color='red', linestyle='dashed')
    plt.plot(effPriorLogOdds, D3, label=l3, color="b")
    plt.plot(effPriorLogOdds, D4, label=l4, color='b', linestyle='dashed')
    if (D5 and D6 and l5 and l6) is not None:
        plt.plot(effPriorLogOdds, D5, label=l5, color="g")
        plt.plot(effPriorLogOdds, D6, label=l6, color="g", linestyle='dashed')
    plt.ylim([0, 1.1])
    plt.xlim([-3, 3])
    plt.legend()
    plt.xlabel(r"$\tilde{p}$")
    plt.ylabel("DCF")
    plt.savefig("../Images/" + figName + ".png")



if __name__ == "__main__":
    # Training and test data, raw features
    DTR, LTR = load("../Data/Train.txt")
    DTE, LTE = load("../Data/Test.txt")
    # Z-score normalization
    DTR_Z = Z_score(DTR)
    DTE_Z = Z_score(DTR, DTE)
    # Gaussianization
    DTR_G = np.load("../Data/gaus_data/gaus_train.npy")
    DTE_G = np.load("../Data/gaus_data/gaus_test.npy")
    # 10-PCA, Gaussianized features
    DTR_G_10 = compute_pca(DTR_G, 10)
    DTE_G_10 = compute_pca(DTR_G, 10, DTE_G)
    # 9-PCA, Gaussianized features
    DTR_G_9 = compute_pca(DTR_G, 9)
    DTE_G_9 = compute_pca(DTR_G, 9, DTE_G)
    # 10-PCA, Z-normalized features
    DTR_Z_10 = compute_pca(DTR_Z, 10)
    DTE_Z_10 = compute_pca(DTR_Z, 10, DTE_Z)

    # Parameters
    pi = 0.5
    pi_T = 0.5
    Cfn = 1
    Cfp = 1
    K_SVM = 1
    C_val = [1e-1, 1, 10]

    ## Gaussian Models
    # RAW
    with open("../Results/gaussian_evaluation.txt", "w") as f_in:
        print("--------- RAW ---------", file=f_in)
        llr_gauss = compute_MVG(DTR, LTR, DTE)
        minDCF_gauss, _ = compute_min_DCF(llr_gauss, LTE, pi, Cfn, Cfp)
        print(str(round(minDCF_gauss, 3)), file=f_in)

        llr_bayes = compute_Naive_Bayes(DTR, LTR, DTE)
        minDCF_bayes, _ = compute_min_DCF(llr_bayes, LTE, pi, Cfn, Cfp)
        print(str(round(minDCF_bayes, 3)), file=f_in)

        llr_tied = compute_Tied(DTR, LTR, DTE)
        minDCF_tied, _ = compute_min_DCF(llr_tied, LTE, pi, Cfn, Cfp)
        print(str(round(minDCF_tied, 3)), file=f_in)

        llr_tied_bayes = compute_Naive_Tied(DTR, LTR, DTE)
        minDCF_tied_bayes, _ = compute_min_DCF(llr_tied_bayes, LTE, pi, Cfn, Cfp)
        print(str(round(minDCF_tied_bayes, 3)), file=f_in)

        # Z - no PCA
        print("--------- Z - NO PCA ---------", file=f_in)
        llr_gauss = compute_MVG(DTR_Z, LTR, DTE_Z)
        minDCF_gauss, _ = compute_min_DCF(llr_gauss, LTE, pi, Cfn, Cfp)
        print(str(round(minDCF_gauss, 3)), file=f_in)

        llr_bayes = compute_Naive_Bayes(DTR_Z, LTR, DTE_Z)
        minDCF_bayes, _ = compute_min_DCF(llr_bayes, LTE, pi, Cfn, Cfp)
        print(str(round(minDCF_bayes, 3)), file=f_in)

        llr_tied = compute_Tied(DTR_Z, LTR, DTE_Z)
        minDCF_tied, _ = compute_min_DCF(llr_tied, LTE, pi, Cfn, Cfp)
        print(str(round(minDCF_tied, 3)), file=f_in)

        llr_tied_bayes = compute_Naive_Tied(DTR_Z, LTR, DTE_Z)
        minDCF_tied_bayes, _ = compute_min_DCF(llr_tied_bayes, LTE, pi, Cfn, Cfp)
        print(str(round(minDCF_tied_bayes, 3)), file=f_in)

        # Gauss - no PCA
        print("--------- G - NO PCA ---------", file=f_in)
        llr_gauss = compute_MVG(DTR_G, LTR, DTE_G)
        minDCF_gauss, _ = compute_min_DCF(llr_gauss, LTE, pi, Cfn, Cfp)
        print(str(round(minDCF_gauss, 3)), file=f_in)

        llr_bayes = compute_Naive_Bayes(DTR_G, LTR, DTE_G)
        minDCF_bayes, _ = compute_min_DCF(llr_bayes, LTE, pi, Cfn, Cfp)
        print(str(round(minDCF_bayes, 3)), file=f_in)

        llr_tied = compute_Tied(DTR_G, LTR, DTE_G)
        minDCF_tied, _ = compute_min_DCF(llr_tied, LTE, pi, Cfn, Cfp)
        print(str(round(minDCF_tied, 3)), file=f_in)

        llr_tied_bayes = compute_Naive_Tied(DTR_G, LTR, DTE_G)
        minDCF_tied_bayes, _ = compute_min_DCF(llr_tied_bayes, LTE, pi, Cfn, Cfp)
        print(str(round(minDCF_tied_bayes, 3)), file=f_in)

        # Gauss - PCA = 10
        print("--------- G - 10 PCA ---------", file=f_in)
        llr_gauss = compute_MVG(DTR_G_10, LTR, DTE_G_10)
        minDCF_gauss, _ = compute_min_DCF(llr_gauss, LTE, pi, Cfn, Cfp)
        print(str(round(minDCF_gauss, 3)), file=f_in)

        llr_bayes = compute_Naive_Bayes(DTR_G_10, LTR, DTE_G_10)
        minDCF_bayes, _ = compute_min_DCF(llr_bayes, LTE, pi, Cfn, Cfp)
        print(str(round(minDCF_bayes, 3)), file=f_in)

        llr_tied = compute_Tied(DTR_G_10, LTR, DTE_G_10)
        minDCF_tied, _ = compute_min_DCF(llr_tied, LTE, pi, Cfn, Cfp)
        print(str(round(minDCF_tied, 3)), file=f_in)

        llr_tied_bayes = compute_Naive_Tied(DTR_G_10, LTR, DTE_G_10)
        minDCF_tied_bayes, _ = compute_min_DCF(llr_tied_bayes, LTE, pi, Cfn, Cfp)
        print(str(round(minDCF_tied_bayes, 3)), file=f_in)

        # Gauss - PCA = 9
        print("--------- G - 9 PCA ---------", file=f_in)
        llr_gauss = compute_MVG(DTR_G_9, LTR, DTE_G_9)
        minDCF_gauss, _ = compute_min_DCF(llr_gauss, LTE, pi, Cfn, Cfp)
        print(str(round(minDCF_gauss, 3)), file=f_in)

        llr_bayes = compute_Naive_Bayes(DTR_G_9, LTR, DTE_G_9)
        minDCF_bayes, _ = compute_min_DCF(llr_bayes, LTE, pi, Cfn, Cfp)
        print(str(round(minDCF_bayes, 3)), file=f_in)

        llr_tied = compute_Tied(DTR_G_9, LTR, DTE_G_9)
        minDCF_tied, _ = compute_min_DCF(llr_tied, LTE, pi, Cfn, Cfp)
        print(str(round(minDCF_tied, 3)), file=f_in)

        llr_tied_bayes = compute_Naive_Tied(DTR_G_9, LTR, DTE_G_9)
        minDCF_tied_bayes, _ = compute_min_DCF(llr_tied_bayes, LTE, pi, Cfn, Cfp)
        print(str(round(minDCF_tied_bayes, 3)), file=f_in)

        # Z - PCA = 10
        print("--------- Z - 10 PCA ---------", file=f_in)
        llr_gauss = compute_MVG(DTR_Z_10, LTR, DTE_Z_10)
        minDCF_gauss, _ = compute_min_DCF(llr_gauss, LTE, pi, Cfn, Cfp)
        print(str(round(minDCF_gauss, 3)), file=f_in)

        llr_bayes = compute_Naive_Bayes(DTR_Z_10, LTR, DTE_Z_10)
        minDCF_bayes, _ = compute_min_DCF(llr_bayes, LTE, pi, Cfn, Cfp)
        print(str(round(minDCF_bayes, 3)), file=f_in)

        llr_tied = compute_Tied(DTR_Z_10, LTR, DTE_Z_10)
        minDCF_tied, _ = compute_min_DCF(llr_tied, LTE, pi, Cfn, Cfp)
        print(str(round(minDCF_tied, 3)), file=f_in)

        llr_tied_bayes = compute_Naive_Tied(DTR_Z_10, LTR, DTE_Z_10)
        minDCF_tied_bayes, _ = compute_min_DCF(llr_tied_bayes, LTE, pi, Cfn, Cfp)
        print(str(round(minDCF_tied_bayes, 3)), file=f_in)
    print("GAUSSIAN \t\t\t DONE")

    ## Logistic Regression
    with open("../Results/linear_regression_evaluation.txt", "w") as f_in:
        for l in [0, 1e-6, 1e-4, 1e-2, 1, 100]:
            print("\n********** lambda = ", l, " **********", file=f_in)
            print("--------- RAW ---------", file=f_in)
            # Linear
            llr_gauss = linear_logistic(DTR, LTR, DTE, l, pi_T)
            minDCF_gauss, _ = compute_min_DCF(llr_gauss, LTE, pi, Cfn, Cfp)
            print(str(round(minDCF_gauss, 3)), file=f_in)
            # Quadratic
            llr_bayes = quadratic_logistic(DTR, LTR, DTE, l, pi_T)
            minDCF_bayes, _ = compute_min_DCF(llr_bayes, LTE, pi, Cfn, Cfp)
            print(str(round(minDCF_bayes, 3)), file=f_in)

            # Z - no PCA
            print("--------- Z - NO PCA ---------", file=f_in)
            llr_gauss = linear_logistic(DTR_Z, LTR, DTE_Z, l, pi_T)
            minDCF_gauss, _ = compute_min_DCF(llr_gauss, LTE, pi, Cfn, Cfp)
            print(str(round(minDCF_gauss, 3)), file=f_in)

            llr_bayes = quadratic_logistic(DTR_Z, LTR, DTE_Z, l, pi_T)
            minDCF_bayes, _ = compute_min_DCF(llr_bayes, LTE, pi, Cfn, Cfp)
            print(str(round(minDCF_bayes, 3)), file=f_in)

            # Gauss - no PCA
            print("--------- G - NO PCA ---------", file=f_in)
            llr_gauss = linear_logistic(DTR_G, LTR, DTE_G, l, pi_T)
            minDCF_gauss, _ = compute_min_DCF(llr_gauss, LTE, pi, Cfn, Cfp)
            print(str(round(minDCF_gauss, 3)), file=f_in)

            llr_bayes = quadratic_logistic(DTR_G, LTR, DTE_G, l, pi_T)
            minDCF_bayes, _ = compute_min_DCF(llr_bayes, LTE, pi, Cfn, Cfp)
            print(str(round(minDCF_bayes, 3)), file=f_in)

            # Gauss - PCA = 10
            print("--------- G - 10 PCA ---------", file=f_in)
            llr_gauss = linear_logistic(DTR_G_10, LTR, DTE_G_10, l, pi_T)
            minDCF_gauss, _ = compute_min_DCF(llr_gauss, LTE, pi, Cfn, Cfp)
            print(str(round(minDCF_gauss, 3)), file=f_in)

            llr_bayes = quadratic_logistic(DTR_G_10, LTR, DTE_G_10, l, pi_T)
            minDCF_bayes, _ = compute_min_DCF(llr_bayes, LTE, pi, Cfn, Cfp)
            print(str(round(minDCF_bayes, 3)), file=f_in)

            # Gauss - PCA = 9
            print("--------- G - 9 PCA ---------", file=f_in)
            llr_gauss = linear_logistic(DTR_G_9, LTR, DTE_G_9, l, pi_T)
            minDCF_gauss, _ = compute_min_DCF(llr_gauss, LTE, pi, Cfn, Cfp)
            print(str(round(minDCF_gauss, 3)), file=f_in)

            llr_bayes = quadratic_logistic(DTR_G_9, LTR, DTE_G_9, l, pi_T)
            minDCF_bayes, _ = compute_min_DCF(llr_bayes, LTE, pi, Cfn, Cfp)
            print(str(round(minDCF_bayes, 3)), file=f_in)

            # Z - PCA = 10
            print("--------- Z - 10 PCA ---------", file=f_in)
            llr_gauss = linear_logistic(DTR_Z_10, LTR, DTE_Z_10, l, pi_T)
            minDCF_gauss, _ = compute_min_DCF(llr_gauss, LTE, pi, Cfn, Cfp)
            print(str(round(minDCF_gauss, 3)), file=f_in)

            llr_bayes = quadratic_logistic(DTR_Z_10, LTR, DTE_Z_10, l, pi_T)
            minDCF_bayes, _ = compute_min_DCF(llr_bayes, LTE, pi, Cfn, Cfp)
            print(str(round(minDCF_bayes, 3)), file=f_in)
    print("LOG REGRESSION \t\t\t DONE")

    ## Linear SVM
    with open("../Results/linear_SVM_evaluation.txt", "w") as f_in:
        print("\nC Values: 1e-1, 1, 10", file=f_in)
        for balance in [False, True]:
            print("\n\nBALANCE: " + str(balance) + "\n", file=f_in)
            print("--------- RAW ---------", file=f_in)
            for C in C_val:
                llr_svm = train_SVM(DTR, LTR, DTE, C, K_SVM, pi_T, balance)
                minDCF_svm, _ = compute_min_DCF(llr_svm, LTE, pi, Cfn, Cfp)
                print(str(round(minDCF_svm, 3)) + "\t", file=f_in, end="")
            print("\n--------- Z - NO PCA ---------", file=f_in)
            for C in C_val:
                llr_svm = train_SVM(DTR_Z, LTR, DTE_Z, C, K_SVM, pi_T, balance)
                minDCF_svm, _ = compute_min_DCF(llr_svm, LTE, pi, Cfn, Cfp)
                print(str(round(minDCF_svm, 3)) + "\t", file=f_in, end="")
            print("\n--------- G - NO PCA ---------", file=f_in)
            for C in C_val:
                llr_svm = train_SVM(DTR_G, LTR, DTE_G, C, K_SVM, pi_T, balance)
                minDCF_svm, _ = compute_min_DCF(llr_svm, LTE, pi, Cfn, Cfp)
                print(str(round(minDCF_svm, 3)) + "\t", file=f_in, end="")
            print("\n--------- G - 10 PCA ---------", file=f_in)
            for C in C_val:
                llr_svm = train_SVM(DTR_G_10, LTR, DTE_G_10, C, K_SVM, pi_T, balance)
                minDCF_svm, _ = compute_min_DCF(llr_svm, LTE, pi, Cfn, Cfp)
                print(str(round(minDCF_svm, 3)) + "\t", file=f_in, end="")
            print("\n--------- G - 9 PCA ---------", file=f_in)
            for C in C_val:
                llr_svm = train_SVM(DTR_G_9, LTR, DTE_G_9, C, K_SVM, pi_T, balance)
                minDCF_svm, _ = compute_min_DCF(llr_svm, LTE, pi, Cfn, Cfp)
                print(str(round(minDCF_svm, 3)) + "\t", file=f_in, end="")
            print("\n--------- Z - 10 PCA ---------", file=f_in)
            for C in C_val:
                llr_svm = train_SVM(DTR_Z_10, LTR, DTE_Z_10, C, K_SVM, pi_T, balance)
                minDCF_svm, _ = compute_min_DCF(llr_svm, LTE, pi, Cfn, Cfp)
                print(str(round(minDCF_svm, 3)) + "\t", file=f_in, end="")
    print("LINEAR SVM \t\t\t DONE")

    ## Polynomial SVM
    with open("../Results/polynomial_SVM_evaluation.txt", "w") as f_in:
        print("\nC Values: 1e-1, 1, 10", file=f_in)
        for balance in [False, True]:
            print("\n\nBALANCE: " + str(balance) + "\n", file=f_in)
            print("--------- RAW ---------", file=f_in)
            for C in C_val:
                llr_svm = train_SVM(DTR, LTR, DTE, C, K_SVM, pi_T, balance, kernel_polynomial, [1, 2, K_SVM ** 0.5])
                minDCF_svm, _ = compute_min_DCF(llr_svm, LTE, pi, Cfn, Cfp)
                print(str(round(minDCF_svm, 3)) + "\t", file=f_in, end="")
            print("\n--------- Z - NO PCA ---------", file=f_in)
            for C in C_val:
                llr_svm = train_SVM(DTR_Z, LTR, DTE_Z, C, K_SVM, pi_T, balance, kernel_polynomial, [1, 2, K_SVM ** 0.5])
                minDCF_svm, _ = compute_min_DCF(llr_svm, LTE, pi, Cfn, Cfp)
                print(str(round(minDCF_svm, 3)) + "\t", file=f_in, end="")
            print("\n--------- G - NO PCA ---------", file=f_in)
            for C in C_val:
                llr_svm = train_SVM(DTR_G, LTR, DTE_G, C, K_SVM, pi_T, balance, kernel_polynomial, [1, 2, K_SVM ** 0.5])
                minDCF_svm, _ = compute_min_DCF(llr_svm, LTE, pi, Cfn, Cfp)
                print(str(round(minDCF_svm, 3)) + "\t", file=f_in, end="")
            print("\n--------- G - 10 PCA ---------", file=f_in)
            for C in C_val:
                llr_svm = train_SVM(DTR_G_10, LTR, DTE_G_10, C, K_SVM, pi_T, balance, kernel_polynomial,
                                    [1, 2, K_SVM ** 0.5])
                minDCF_svm, _ = compute_min_DCF(llr_svm, LTE, pi, Cfn, Cfp)
                print(str(round(minDCF_svm, 3)) + "\t", file=f_in, end="")
            print("\n--------- G - 9 PCA ---------", file=f_in)
            for C in C_val:
                llr_svm = train_SVM(DTR_G_9, LTR, DTE_G_9, C, K_SVM, pi_T, balance, kernel_polynomial,
                                    [1, 2, K_SVM ** 0.5])
                minDCF_svm, _ = compute_min_DCF(llr_svm, LTE, pi, Cfn, Cfp)
                print(str(round(minDCF_svm, 3)) + "\t", file=f_in, end="")
            print("\n--------- Z - 10 PCA ---------", file=f_in)
            for C in C_val:
                llr_svm = train_SVM(DTR_Z_10, LTR, DTE_Z_10, C, K_SVM, pi_T, balance, kernel_polynomial,
                                    [1, 2, K_SVM ** 0.5])
                minDCF_svm, _ = compute_min_DCF(llr_svm, LTE, pi, Cfn, Cfp)
                print(str(round(minDCF_svm, 3)) + "\t", file=f_in, end="")
    print("QUADRATIC SVM \t\t\t DONE")

    ## RBG SVM
    with open("../Results/RBG_SVM_evaluation.txt", "w") as f_in:
        print("\nC Values: 1e-1, 1, 10", file=f_in)
        for gamma in [np.exp(-1), np.exp(-2)]:
            print("\n\nGAMMA: " + str(gamma) + "\n", file=f_in)
            print("--------- RAW ---------", file=f_in)
            for C in C_val:
                llr_svm = train_SVM(DTR, LTR, DTE, C, K_SVM, pi_T, True, kernel_rbf, [gamma, K_SVM ** 0.5])
                minDCF_svm, _ = compute_min_DCF(llr_svm, LTE, pi, Cfn, Cfp)
                print(str(round(minDCF_svm, 3)) + "\t", file=f_in, end="")
            print("\n--------- Z - NO PCA - no balance ---------", file=f_in)
            for C in C_val:
                llr_svm = train_SVM(DTR_Z, LTR, DTE_Z, C, K_SVM, pi_T, False, kernel_rbf, [gamma, K_SVM ** 0.5])
                minDCF_svm, _ = compute_min_DCF(llr_svm, LTE, pi, Cfn, Cfp)
                print(str(round(minDCF_svm, 3)) + "\t", file=f_in, end="")
            print("\n--------- Z - NO PCA - balance ---------", file=f_in)
            for C in C_val:
                llr_svm = train_SVM(DTR_Z, LTR, DTE_Z, C, K_SVM, pi_T, True, kernel_rbf, [gamma, K_SVM ** 0.5])
                minDCF_svm, _ = compute_min_DCF(llr_svm, LTE, pi, Cfn, Cfp)
                print(str(round(minDCF_svm, 3)) + "\t", file=f_in, end="")
            print("\n--------- G - NO PCA ---------", file=f_in)
            for C in C_val:
                llr_svm = train_SVM(DTR_G, LTR, DTE_G, C, K_SVM, pi_T, True, kernel_rbf, [gamma, K_SVM ** 0.5])
                minDCF_svm, _ = compute_min_DCF(llr_svm, LTE, pi, Cfn, Cfp)
                print(str(round(minDCF_svm, 3)) + "\t", file=f_in, end="")
    print("RBF SVM \t\t\t DONE")

    ## GMM
    components = [4, 8, 16]
    fileName = "../Results/GMM_results_evaluation.txt"
    with open(fileName, "w") as f_in:
        for component in components:
            print("COMPONENTS: " + str(component), file=f_in)
            for tied in [False, True]:
                print("TIED: " + str(tied), file=f_in)
                for diag in [False, True]:
                    print("DIAG: " + str(diag), file=f_in)
                    llr_gmm = GMM_classifier(DTR, LTR, DTE, 2, component, diag, tied)
                    minDCF_gmm, _ = compute_min_DCF(llr_gmm, LTE, pi, Cfn, Cfp)
                    print("\tRAW - " + str(round(minDCF_gmm, 3)), file=f_in)
                    llr_gmm = GMM_classifier(DTR_Z, LTR, DTE_Z, 2, component, diag, tied)
                    minDCF_gmm, _ = compute_min_DCF(llr_gmm, LTE, pi, Cfn, Cfp)
                    print("\tZ - " + str(round(minDCF_gmm, 3)), file=f_in)
                    llr_gmm = GMM_classifier(DTR_G, LTR, DTE_G, 2, component, diag, tied)
                    minDCF_gmm, _ = compute_min_DCF(llr_gmm, LTE, pi, Cfn, Cfp)
                    print("\tG - " + str(round(minDCF_gmm, 3)), file=f_in)
    print("GMM SVM \t\t\t DONE")

    ## Fusions llr of train set of best models
    llrSVMTrain = np.load("../Data/llrSVM.npy")
    llrLRTrain = np.load("../Data/llrLR.npy")
    llrGMMTrain = np.load("../Data/llrGMM.npy")

    # Train models on all training dataset and get scores on evaluation dataset
    llrSVMTest = train_SVM(DTR_Z, LTR, DTE_Z, 10, K_SVM, pi, True, kernel=kernel_rbf,
                           args_kernel=[np.exp(-2), K_SVM ** 0.5])
    llrLRTest = quadratic_logistic(DTR_Z, LTR, DTE_Z, 0, pi_T)
    llrGMMTest = GMM_classifier(DTR_Z, LTR, DTE_Z, 2, 8, False, False)

    # Save obtained scores
    np.save("../Data/llrSVMTest.npy", llrSVMTest)
    np.save("../Data/llrLRTest.npy", llrLRTest)
    np.save("../Data/llrGMMTest.npy", llrGMMTest)

    # Evaluate performance of combined models
    # (LR model trained on training set scores and evaluated on test set scores)


    fileName = "../Results/fusions_results_eval.txt"
    with open(fileName, "w") as f:
        f.write("----------- Actual DCF of single models -----------*")
        min_DCF = compute_min_DCF(llrSVMTest, LTE, pi, Cfn, Cfp)
        actDCF = compute_act_DCF(llrSVMTest, LTE, pi, Cfn, Cfp)
        actDCF_estimated = evaluate_score_calibration(llrSVMTrain, llrSVMTest, LTR, LTE, pi, Cfn, Cfp)
        f.write("\n\nSVM: min:" + str(min_DCF) + "  actual: " + str(actDCF) + " estimated: " + str(
            actDCF_estimated))

        min_DCF = compute_min_DCF(llrLRTest, LTE, pi, Cfn, Cfp)
        actDCF = compute_act_DCF(llrLRTest, LTE, pi, Cfn, Cfp)
        actDCF_estimated = evaluate_score_calibration(llrLRTrain, llrLRTest, LTR, LTE, pi, Cfn, Cfp)
        f.write(
            "\nLR: min:" + str(min_DCF) + "   actual: " + str(actDCF) + " estimated: " + str(actDCF_estimated))
        actDCF = compute_act_DCF(llrGMMTest, LTE, pi, Cfn, Cfp)
        actDCF_estimated = evaluate_score_calibration(llrGMMTrain, llrGMMTest, LTR, LTE, pi, Cfn, Cfp)
        min_DCF = compute_min_DCF(llrGMMTest, LTE, pi, Cfn, Cfp)

        f.write("\nGMM: min:" + str(min_DCF) + "  actual: " + str(actDCF) + " estimated: " + str(actDCF_estimated))

        opt_lambda_SVMLR = 0.001
        opt_lambda_SVMGMM = 0.001
        opt_lambda_SVMLRGMM = 0

        f.write("\n\n----------- SVM + LR -----------*\n\n")
        minDCF, actDCF, _ = evaluate_2_models_combination(llrSVMTrain, llrLRTrain, llrSVMTest, llrLRTest, LTR, LTE,
                                                          opt_lambda_SVMLR)
        f.write("min DCF: " + str(minDCF) + " act DCF: " + str(actDCF))

        f.write("\n\n----------- SVM + GMM -----------*\n\n")
        minDCF, actDCF, _ = evaluate_2_models_combination(llrSVMTrain, llrGMMTrain, llrSVMTest, llrGMMTest, LTR, LTE,
                                                          opt_lambda_SVMGMM)
        f.write("min DCF: " + str(minDCF) + " act DCF: " + str(actDCF))

        f.write("\n\n----------- SVM + LR + GMM -----------*\n\n")
        minDCF, actDCF, _ = evaluate_3_models_combination(llrSVMTrain, llrLRTrain, llrGMMTrain, llrSVMTest, llrLRTest,
                                                          llrGMMTest, LTR, LTE, opt_lambda_SVMLRGMM)
        f.write("min DCF: " + str(minDCF) + " act DCF: " + str(actDCF))

    ## ROC
    # Compute components of ROC plots
    FPR_SVM, TPR_SVM = ROC_components(llrSVMTest, LTE)
    FPR_LR, TPR_LR = ROC_components(llrLRTest, LTE)
    FPR_GMM, TPR_GMM = ROC_components(llrGMMTest, LTE)

    _, _, llrSVMLR = evaluate_2_models_combination(llrSVMTrain, llrLRTrain, llrSVMTest, llrLRTest, LTR, LTE,
                                                   opt_lambda_SVMLR)
    _, _, llrSVMGMM = evaluate_2_models_combination(llrSVMTrain, llrGMMTrain, llrSVMTest, llrGMMTest, LTR, LTE,
                                                    opt_lambda_SVMGMM)
    _, _, llrSVMLRGMM = evaluate_3_models_combination(llrSVMTrain, llrLRTrain, llrGMMTrain, llrSVMTest, llrLRTest,
                                                      llrGMMTest,
                                                      LTR, LTE, opt_lambda_SVMLRGMM)

    FPR_SVMLR, TPR_SVMLR = ROC_components(llrSVMLR, LTE)
    FPR_SVMGMM, TPR_SVMGMM = ROC_components(llrSVMGMM, LTE)
    FPR_SVMLRGMM, TPR_SVMLRGMM = ROC_components(llrSVMLRGMM, LTE)

    # Save the components of ROCs
    np.save("../Data/FPR_SVM.npy", FPR_SVM)
    np.save("../Data/TPR_SVM.npy", TPR_SVM)
    np.save("../Data/FPR_LR.npy", FPR_LR)
    np.save("../Data/TPR_LR.npy", TPR_LR)
    np.save("../Data/FPR_GMM.npy", FPR_GMM)
    np.save("../Data/TPR_GMM.npy", TPR_GMM)
    np.save("../Data/FPR_SVMLR.npy", FPR_SVMLR)
    np.save("../Data/TPR_SVMLR.npy", TPR_SVMLR)
    np.save("../Data/FPR_SVMGMM.npy", FPR_SVMGMM)
    np.save("../Data/TPR_SVMGMM.npy", TPR_SVMGMM)
    np.save("../Data/FPR_SVMLRGMM.npy", FPR_SVMLRGMM)
    np.save("../Data/TPR_SVMLRGMM.npy", TPR_SVMLRGMM)

    # Plot the ROC curves
    plot_ROC_curve(FPR_SVM, TPR_SVM, FPR_LR, TPR_LR, FPR_GMM, TPR_GMM, "SVM", "LR", "GMM", "ROC_eval1")
    plot_ROC_curve(FPR_SVMLR, TPR_SVMLR, FPR_SVMGMM, TPR_SVMGMM, FPR_SVMLRGMM, TPR_SVMLRGMM, "SVM+LR", "SVM+GMM",
                   "SVM+LR+GMM", "ROC_eval2")
    plot_ROC_curve(FPR_SVM, TPR_SVM, FPR_LR, TPR_LR, FPR_SVMLRGMM, TPR_SVMLRGMM, "SVM", "LR", "SVM+LR+GMM", "ROC_eval3")

    ## Bayes Error Plots
    # Compute needed components
    DCF_SVM, minDCF_SVM = Bayes_plot_components(llrSVMTest, LTE)
    DCF_LR, minDCF_LR = Bayes_plot_components(llrLRTest, LTE)
    DCF_GMM, minDCF_GMM = Bayes_plot_components(llrGMMTest, LTE)
    DCF_SVMLR, minDCF_SVMLR = Bayes_plot_components(llrSVMLR, LTE)
    DCF_SVMGMM, minDCF_SVMGMM = Bayes_plot_components(llrSVMGMM, LTE)
    DCF_SVMLRGMM, minDCF_SVMLRGMM = Bayes_plot_components(llrSVMLRGMM, LTE)

    # Save Bayes plot components
    np.save("../Data/DCF_SVM.npy", DCF_SVM)
    np.save("../Data/minDCF_SVM.npy", minDCF_SVM)
    np.save("../Data/DCF_LR.npy", DCF_LR)
    np.save("../Data/minDCF_LR.npy", minDCF_LR)
    np.save("../Data/DCF_GMM.npy", DCF_GMM)
    np.save("../Data/minDCF_GMM.npy", minDCF_GMM)
    np.save("../Data/DCF_SVMLR.npy", DCF_SVMLR)
    np.save("../Data/minDCF_SVMLR.npy", minDCF_SVMLR)
    np.save("../Data/DCF_SVMGMM.npy", DCF_SVMGMM)
    np.save("../Data/minDCF_SVMGMM.npy", minDCF_SVMGMM)
    np.save("../Data/DCF_SVMLRGMM.npy", DCF_SVMLRGMM)
    np.save("../Data/minDCF_SVMLRGMM.npy", minDCF_SVMLRGMM)

    # Plot bayes error graphs
    plot_Bayes_error(DCF_SVM, minDCF_SVM, DCF_LR, minDCF_LR,
                     "SVM: act DCF", "SVM: min DCF", "LR: act DCF", "LR: min DCF",
                     "Bayes_error1_eval", DCF_GMM, minDCF_GMM, "GMM: act DCF", "GMM: min DCF")
    plot_Bayes_error(DCF_SVMLR, minDCF_SVMLR, DCF_SVMGMM, minDCF_SVMGMM,
                     "SVM+LR: act DCF", "SVM+LR: min DCF", "SVM+GMM: act DCF", "SVM+GMM: min DCF","Bayes_error2_eval",
                     DCF_SVMLRGMM, minDCF_SVMLRGMM, "SVM+LR+GMM: act DCF", "SVM+LR+GMM: min DCF")
    plot_Bayes_error(DCF_SVM, DCF_LR, DCF_SVMLRGMM, minDCF_SVMLRGMM,
                      "SVM: act DCF", "LR: act DCF", "SVM+LR+GMM: act DCF", "SVM+LR+GMM: min DCF", "Bayes_error3_eval")
