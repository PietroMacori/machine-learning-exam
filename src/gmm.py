# GAUSSIAN MIXTURE MODELS
#
# This file contains the functions needed implement a Gaussian Mixture Model classifier. It is based on the
# LBG and Expectation-Maximization algorithms to iteratively compute the gaussian composing the GMM and their
# optimal parameters. The K-fold approach has been used to validate the model and estimate the best number of
# components to be used. The performance of the models are evaluated using the minimum DCF function.

import numpy as np
from load_data import load
import scipy
from DCF import compute_min_DCF
from preprocessing import Z_score
import matplotlib.pyplot as plt


# Function to compute log-likelihood ratio given log-likelihood
def compute_llr(ll):
    if ll.shape[0] != 2:
        return 0
    # Since in log domain - subtract the ll of the two classes
    return ll[1, :] - ll[0, :]


# Compute log-likelihood of samples over a gaussian model
def GAU_logpdf_ND(X, mu, C):
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


# Compute log-likelihood of set of samples using a GMM
def logpdf_GMM(X, gmm):
    M = len(gmm)
    N = X.shape[1]

    # Define S matrix to store cluster conditional densities
    S = np.zeros([M, N])

    for g in range(M):
        # Class joint log-density + prior (weight)
        S[g, :] = GAU_logpdf_ND(X, gmm[g][1], gmm[g][2]) + np.log(gmm[g][0])

    # Marginal log densities
    log_dens = scipy.special.logsumexp(S, axis=0)
    # Posterior distributions
    Post = np.exp(S - log_dens)
    return log_dens, Post


# Implementation of the Expectation-Maximization algorithm
def GMM_EM_estimation(X, gmm, threshold, psi, diag=False, tied=False):
    curr_gmm = gmm
    ll = threshold + 1
    prev_ll = 0

    # Stop condition on log-likelihood variation
    while abs(ll - prev_ll) >= threshold:
        # E-step: compute posterior probabilities
        logdens, gamma = logpdf_GMM(X, curr_gmm)
        if prev_ll == 0:
            prev_ll = sum(logdens) / X.shape[1]
        else:
            prev_ll = ll
        # M-step: update model parameters
        Z = np.sum(gamma, axis=1)

        for g in range(len(gmm)):
            # Compute statistics
            F = np.sum(gamma[g] * X, axis=1)
            S = np.dot(gamma[g] * X, X.T)
            mu = (F / Z[g]).reshape([X.shape[0], 1])
            sigma = S / Z[g] - np.dot(mu, mu.T)
            w = Z[g] / sum(Z)

            if diag:
                # Keep only the diagonal of the matrix
                sigma = sigma * np.eye(sigma.shape[0])

            if not tied:  # If tied hypothesis, add constraints only at the end
                U, s, _ = np.linalg.svd(sigma)
                # Add constraints on the covariance matrixes to avoid degenerate solutions
                s[s < psi] = psi
                covNew = np.dot(U, s.reshape([s.shape[0], 1]) * U.T)
                curr_gmm[g] = (w, mu, covNew)
            else:  # if tied, constraints are added later
                curr_gmm[g] = (w, mu, sigma)

        if tied:
            # Compute tied covariance matrix
            tot_sigma = np.zeros(curr_gmm[0][2].shape)
            for g in range(len(gmm)):
                tot_sigma += Z[g] * curr_gmm[g][2]
            tot_sigma /= X.shape[1]
            U, s, _ = np.linalg.svd(tot_sigma)
            # Add constraints on the covariance matrixes to avoid degenerate solutions
            s[s < psi] = psi
            tot_sigma = np.dot(U, s.reshape([s.shape[0], 1]) * U.T)
            for g in range(len(gmm)):
                curr_gmm[g][2][:, :] = tot_sigma

                # Compute log-likelihood of training data
        logdens, _ = logpdf_GMM(X, curr_gmm)
        ll = sum(logdens) / X.shape[1]

    return curr_gmm


# Implement LBG algorithm
def LBG(X, gmm, th, alpha, psi, diag, tied):
    new_gmm = []
    for c in gmm:
        # Find eigenvalue corresponding to the largest eigenvalue of the cov matrix
        U, s, _ = np.linalg.svd(c[2])
        # Compute displacement of the mean
        d = U[:, 0:1] * s[0] ** 0.5 * alpha
        new_gmm.append((c[0] / 2, c[1].reshape([X.shape[0], 1]) + d, c[2]))
        new_gmm.append((c[0] / 2, c[1].reshape([X.shape[0], 1]) - d, c[2]))

    # Tune parameters using EM
    gmm = GMM_EM_estimation(X, new_gmm, th, psi, diag, tied)

    return gmm


# Implementation of the Gaussian Mixture Model classifier
def GMM_classifier(DTR, LTR, DTE, n_classes, num_components, diag, tied, t=1e-6, psi=0.01, alpha=0.1):
    S = np.zeros([n_classes, DTE.shape[1]])
    all_gmm = []
    llr = 0

    for count in range(int(np.log2(num_components))):
        for c in range(n_classes):
            # Define the starting components of gmm
            if count == 0:
                # Manage first iteration
                covNew = np.cov(DTR[:, LTR == c])
                # Impose the constraint on the covariance matrix
                U, s, _ = np.linalg.svd(covNew)
                s[s < psi] = psi
                covNew = np.dot(U, s.reshape([s.shape[0], 1]) * U.T)
                # Start from max likelihood solution for one component
                starting_gmm = [(1.0, np.mean(DTR[:, LTR == c], axis=1), covNew)]
                all_gmm.append(starting_gmm)
            else:
                starting_gmm = all_gmm[c]

            new_gmm = LBG(DTR[:, LTR == c], starting_gmm, t, alpha, psi, diag, tied)
            all_gmm[c] = new_gmm
            log_densities, _ = logpdf_GMM(DTE, new_gmm)
            S[c, :] = log_densities

        llr = compute_llr(S)
    return llr


# Perform k-fold cross validation on test data for the specified model
def k_fold(Data, Labels, K, pi, Cfp, Cfn, diag, tied, components, seed=0):
    # Number of samples of a single fold
    fold_dim = Data.shape[1] // K
    start_idx = 0

    # Shuffle data
    np.random.seed(seed)
    idx = np.random.permutation(Data.shape[1])

    # Define array as long as the training data since with K fold all samples are used to test the model
    llr = np.zeros((Data.shape[1],))

    for i in range(K):
        print("A", i)
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
        llr[idxTest] = GMM_classifier(DTR, LTR, DTE, 2, components, diag, tied)

        start_idx += fold_dim

    # Compute minimum DCF
    minDCF, _ = compute_min_DCF(llr, Labels, pi, Cfn, Cfp)

    return minDCF, llr


if __name__ == "__main__":
    # Load data
    D, L = load("../Data/Train.txt")
    DN = Z_score(D)
    DG = np.load("../Data/gaus_data/gaus_train.npy")

    # Define parameters
    k = 5
    pi = 0.5
    Cfn = 1
    Cfp = 1
    components_val = [2, 4, 8, 16]
    fileName = "../Results/GMM_results.txt"

    # Define matrix for storing results for various components
    DCF_z = np.zeros([4, len(components_val)])
    DCF_gaus = np.zeros([4, len(components_val)])

    with open(fileName, "w") as f:
        f.write("\nTied      \tDiag      \tComponents\tMinDFC    \n")
        for i, tied in enumerate([False, True]):
            for j, diag in enumerate([False, True]):
                for m, components in enumerate(components_val):
                    f.write("\n Tied: " + str(tied) + " Diag: " + str(diag) + " components: " + str(components))
                    minDCF, _ = k_fold(DN, L, k, pi, Cfp, Cfn, diag, tied, components)
                    DCF_z[2 * i + j, m] = minDCF
                    f.write("\nZ-norm\n" + str(tied) + "     \t" + str(diag) + "     \t" + str(
                        components) + "         \t" + str(minDCF) + "\n")
                    minDCF, _ = k_fold(DG, L, k, pi, Cfp, Cfn, diag, tied, components)
                    DCF_gaus[2 * i + j, m] = minDCF
                    f.write("\nGaussianized\n" + str(tied) + "     \t" + str(diag) + "     \t" + str(
                        components) + "     \t" + str(round(minDCF, 3)) + "\n\n")
                    print("Tied: %s, diag: %s, components: %d" % (str(tied), str(diag), components))

    # Plot and save min DCF graph for all combinations of tied and diag
    for i, imgName in enumerate(["GMM_tied_nodiag", "GMM_tied_diag", "GMM_notied_nodiag", "GMM_notied_diag"]):
        plt.figure()
        plt.plot(components_val, DCF_z[i, :], marker='o', linestyle='dashed', color="red")
        plt.plot(components_val, DCF_gaus[i, :], marker='o', linestyle='dashed', color="blue")
        plt.xlabel("Components")
        plt.ylabel("min DCF")
        plt.legend(["Z-normalized", "Gaussianized"])
        plt.savefig("../Images/" + imgName)
