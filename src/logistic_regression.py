# LOGISTIC REGRESSION MODELS
#
# This file contains the functions needed to train and test 2 logistic regression classifiers: linear and quadratic.
# Besides the two classifiers, the k-fold function is implemented to cross validate the implemented models as well as
# the objective function, which is exploited by the numeric solver to find the optimal w and b.
# Differently pre-processed data, as well as raw data are used to train the model. Different values of lambda are tested
# using cross validation. Results are stored in files and plots of minDCF vs lambda are saved.


import numpy as np
import scipy
from DCF import compute_min_DCF
import matplotlib.pyplot as plt
from pca import compute_pca
from preprocessing import Z_score
from load_data import load


# Compute objective function and its derivatives, use wrapper to pass args
def logreg_obj_wrap(DTR, LTR, l, pi_T):
    def logreg_obj(v):
        # Extract w and b
        w, b = v[0:-1], v[-1]

        # Compute number of samples of the two classes
        Nt = sum(LTR == 1)
        Nf = sum(LTR == 0)

        # Compute Objective Function
        R = l / 2 * np.linalg.norm(w) ** 2 + pi_T / Nt * sum(np.log1p(np.exp(- (np.dot(w.T, DTR[:, LTR == 1]) + b)))) + \
            (1 - pi_T) / Nf * sum(np.log1p(np.exp((np.dot(w.T, DTR[:, LTR == 0]) + b))))

        # Compute partial derivative wrt w
        dR_w = l * w - pi_T / Nt * np.sum(DTR[:, LTR == 1] / (1 + np.exp(np.dot(w.T, DTR[:, LTR == 1]) + b)), axis=1) + \
              (1 - pi_T) / Nf * np.sum(DTR[:, LTR == 0] / (1 + np.exp(-np.dot(w.T, DTR[:, LTR == 0]) - b)), axis=1)

        # Compute partial derivative wrt b
        dR_b = -pi_T / Nt * np.sum(1 / (1 + np.exp(np.dot(w.T, DTR[:, LTR == 1]) + b))) + \
              (1 - pi_T) / Nf * np.sum(1 / (1 + np.exp(-np.dot(w.T, DTR[:, LTR == 0]) - b)))

        # Gradient of the Objective Function
        dR = np.concatenate((dR_w, np.array(dR_b).reshape(1, )))

        return R, dR

    return logreg_obj


# Function implementing the Linear Logistic Model. Requires the lambda for regularization term and pi_T for balancing
def linear_logistic(DTR, LTR, DTE, l, pi_T):
    # Define objective function
    # NOTE: use wrapper function to pass arguments (bfgs accepts only one array)
    logreg_obj = logreg_obj_wrap(DTR, LTR, l, pi_T)

    # Optimize objective function, returns the 
    v_star, _, _ = scipy.optimize.fmin_l_bfgs_b(logreg_obj, np.zeros(DTR.shape[0] + 1), approx_grad=False)

    # Retrieve optimal w and b
    w_opt = v_star[0:-1]
    b_opt = v_star[-1]

    # Compute the scores -> wT * x + b
    s = np.dot(w_opt.T, DTE) + b_opt

    return s


# Function implementing the Quadratic Logistic Model.
def quadratic_logistic(DTR, LTR, DTE, l, pi_T):
    # phi will contain concatenation of vec(x*xT) and x
    phi = np.zeros([DTR.shape[0] ** 2 + DTR.shape[0], DTR.shape[1]])

    # Map train samples to expanded feature space
    for index in range(DTR.shape[1]):
        x = DTR[:, index].reshape(DTR.shape[0], 1)
        vec = np.dot(x, x.T).reshape(x.shape[0] ** 2, 1)
        phi[:, index] = np.concatenate((vec, x)).reshape(phi.shape[0], )

    # Define objective function
    # NOTE: phi is passed to the function
    logreg_obj = logreg_obj_wrap(phi, LTR, l, pi_T)
    # Optimize objective function
    v_star, _, _ = scipy.optimize.fmin_l_bfgs_b(logreg_obj, np.zeros(phi.shape[0] + 1), approx_grad=False)

    # Retrieve optimal w and b
    w_opt = v_star[0:-1]
    b_opt = v_star[-1]

    # Map test samples to expanded feature space
    phi_test = np.zeros([DTE.shape[0] ** 2 + DTE.shape[0], DTE.shape[1]])
    for index in range(DTE.shape[1]):
        x_test = DTE[:, index].reshape(DTE.shape[0], 1)
        vec_test = np.dot(x_test, x_test.T).reshape(x_test.shape[0] ** 2, 1)
        phi_test[:, index] = np.concatenate((vec_test, x_test)).reshape(phi_test.shape[0], )

    # Compute the scores using phi_test
    s = np.dot(w_opt.T, phi_test) + b_opt

    return s


# Perform k-fold cross validation on test data for the specified model
def k_fold(Data, Labels, classifier, K, pi, Cfp, Cfn, l, pi_T, seed=0):
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
        llr[idxTest] = classifier(DTR, LTR, DTE, l, pi_T)
        start_idx += fold_dim

    minDCF, _ = compute_min_DCF(llr, Labels, pi, Cfn, Cfp)

    return minDCF, llr


if __name__ == "__main__":
    # Load data
    D, L = load("../Data/Train.txt")
    DN = Z_score(D)
    DN_10 = compute_pca(DN, 10)
    DG = np.load("../Data/gaus_data/gaus_train.npy")
    DG_10 = compute_pca(DG, 10)

    # Parameters
    l_val = [0, 1e-6, 1e-4, 1e-2, 1, 100]  # Evaluate best lambda using cross validation
    pi_T = 0.5
    pi = 0.5
    Cfn = 1
    Cfp = 1
    k = 5

    # Iterate on both linear and quadratic
    for LR_type in [linear_logistic, quadratic_logistic]:
        if LR_type == linear_logistic:
            img = "Linear_regression_lambda.png"
            fileName = "../Results/linear_regression_results.txt"
        else:
            img = "Quadratic_regression_lambda.png"
            fileName = "../Results/quadratic_regression_results.txt"

        with open(fileName, "w") as f:
            # Raw
            f.write("lambda values\n 0   \t1e-6 \t1e-4 \t1e-2 \t1    \t100  \n")
            f.write("\nRaw\n")
            DCF_raw = []
            for l in l_val:
                minDCF, _ = k_fold(D, L, LR_type, k, pi, Cfp, Cfn, l, pi_T)
                DCF_raw.append(minDCF)
                f.write(str(round(minDCF, 3)) + "\t")
            # Z-normalized - no PCA
            f.write("\n\nZ-normalized - no PCA\n")
            DCF_z = []
            for l in l_val:
                minDCF, _ = k_fold(DN, L, LR_type, k, pi, Cfp, Cfn, l, pi_T)
                DCF_z.append(minDCF)
                f.write(str(round(minDCF, 3)) + "\t")

            # Z-normalized - PCA = 10
            f.write("\n\nZ-normalized - PCA = 10\n")
            DCF_z_10 = []
            for l in l_val:
                minDCF, _ = k_fold(DN_10, L, LR_type, k, pi, Cfp, Cfn, l, pi_T)
                DCF_z_10.append(minDCF)
                f.write(str(round(minDCF, 3)) + "\t")

            # Gaussianized - no PCA
            f.write("\n\nGaussianized\n")
            DCF_gauss = []
            for l in l_val:
                minDCF, _ = k_fold(DG, L, LR_type, k, pi, Cfp, Cfn, l, pi_T)
                DCF_gauss.append(minDCF)
                f.write(str(round(minDCF, 3)) + "\t")

            # Gaussianized - PCA = 10
            f.write("\n\nGaussianized - PCA = 10\n")
            DCF_gauss_10 = []
            for l in l_val:
                minDCF, _ = k_fold(DG_10, L, LR_type, k, pi, Cfp, Cfn, l, pi_T)
                DCF_gauss_10.append(minDCF)
                f.write(str(round(minDCF, 3)) + "\t")

            # Plot min DCF for different values of lambda
            plt.figure()
            plt.plot(l_val, DCF_raw)
            plt.plot(l_val, DCF_z_10)
            plt.plot(l_val, DCF_gauss)
            plt.xscale("log")
            plt.xlabel(r"$\lambda$")
            plt.ylabel("min DCF")
            plt.legend(["Raw", "Z-normalized", "Gaussianized"])
            plt.savefig("../Images/" + img)
