# PRINCIPAL COMPONENT ANALYSIS
#
# This file contains the function needed to compute the PCA over a dataset.
# It allows to compute the PCA over training and testing data.
# It has been decided to use the eigen-decomposition method over the singular value decomposition
# for simplicity and readability of the code.

import numpy as np
from utility import vcol


# Compute the Principal Component Analysis
def compute_pca(data, num_feat, data_eval=None):
    # Average col vect
    mu = vcol(data.mean(1))

    # Centered data
    DC = data - mu

    # Covariance matrix
    C = 1 / DC.shape[1] * np.dot(DC, DC.T)

    # Eigenvalues and eigenvectors
    s, U = np.linalg.eigh(C)

    # Get the P matrix composed by eigenvectors of the largest eigenvalues
    P = U[:, ::-1][:, 0: num_feat]

    if data_eval is None:
        # Projection of train data
        DP = np.dot(P.T, data)
    else:
        # Projection of test data
        DP = np.dot(P.T, data_eval)

    return DP
