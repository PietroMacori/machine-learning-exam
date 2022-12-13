# DETECTION COST FUNCTION
#
# This file contains the functions needed to compute the minimum and actual DCF for the various classifiers.
# This provides an objective metric to compare the various classifiers. Moreover, it contains functions to
# compute the components of the ROC and Bayes Error plot.

import numpy as np


# Expectation of Bayes cost using optimal Bayes decision
def bayes_risk(CM, pi, Cfn, Cfp):
    FNR = CM[0, 1] / (CM[0, 1] + CM[1, 1])
    FPR = CM[1, 0] / (CM[1, 0] + CM[0, 0])
    return pi * Cfn * FNR + (1 - pi) * Cfp * FPR


# Normalized Bayes risk wrt best dummy case
def norm_bayes_risk(CM, pi, Cfn, Cfp):
    br = bayes_risk(CM, pi, Cfn, Cfp)
    B_dummy = min(pi * Cfn, (1 - pi) * Cfp)
    return br / B_dummy


# Predict labels depending on threshold
def optimal_Bayes_prediction(llr, pi, Cfn, Cfp, th=None):
    if th is None:
        th = - np.log((pi * Cfn) / ((1 - pi) * Cfp))

    Pred = np.zeros([llr.shape[0]])
    Pred[llr > th] = 1

    return Pred


# Compute Confusion Matrix given predictions and actual labels
def compute_binary_confusion(predictions, labels):
    CM = np.zeros((2, 2))
    CM[0, 0] = ((predictions == 0) * (labels == 0)).sum()
    CM[0, 1] = ((predictions == 0) * (labels == 1)).sum()
    CM[1, 0] = ((predictions == 1) * (labels == 0)).sum()
    CM[1, 1] = ((predictions == 1) * (labels == 1)).sum()
    return CM


# Compute the actual Detection Cost Function
def compute_act_DCF(scores, labels, pi, Cfn, Cfp, th=None):
    predictions = optimal_Bayes_prediction(scores, pi, Cfn, Cfp, th=th)
    CM = compute_binary_confusion(predictions, labels)
    return norm_bayes_risk(CM, pi, Cfn, Cfp)


# Compute the minimum Detection Cost Function
def compute_min_DCF(scores, labels, pi, Cfn, Cfp):
    # Define array of thresholds composed by the received scores
    t = np.array(scores).reshape(scores.size, )
    t.sort()
    # Include -inf and +inf
    t = np.concatenate([np.array([-np.inf]), t, np.array([+np.inf])])
    dcfList = list()
    # For each threshold compute the actual DCF
    for th in t:
        dcfList.append(compute_act_DCF(scores, labels, pi, Cfn, Cfp, th=th))

    # Retrieve the minimum DCF computed
    minDCF = np.array(dcfList).min()
    minIdx = np.argmin(dcfList)
    # Retrieve the corresponding threshold
    optThreshold = t[minIdx]

    return minDCF, optThreshold


# Plot the Receiver Operating Characteristic
def ROC_components(scores, labels):
    t = np.array(scores).reshape(scores.size, )
    t.sort()
    # Include -inf and +inf
    t = np.concatenate([np.array([-np.inf]), t, np.array([+np.inf])])

    # Store the FPR and TPR in arrays
    FPR = np.zeros([t.shape[0]])
    TPR = np.zeros([t.shape[0]])
    # For each threshold compute the FPR and TPR
    for index, t in enumerate(t):
        PredictedLabels = np.zeros([scores.shape[0]])
        PredictedLabels[scores > t] = 1
        M = compute_binary_confusion(PredictedLabels, labels)
        FNR = M[0, 1] / (M[0, 1] + M[1, 1])
        FPR[index] = M[1, 0] / (M[0, 0] + M[1, 0])
        TPR[index] = 1 - FNR

    return FPR, TPR


# Calculate data to print Bayes error plots
def Bayes_plot_components(llr, true_labels):
    effPriorLogOdds = np.linspace(-3, 3, 21)
    DCF = np.zeros([effPriorLogOdds.shape[0]])
    minDCF = np.zeros([effPriorLogOdds.shape[0]])

    for index, p_tilde in enumerate(effPriorLogOdds):
        pi_tilde = 1 / (1 + np.exp(-p_tilde))
        pred = optimal_Bayes_prediction(llr, pi_tilde, 1, 1)
        M = compute_binary_confusion(pred, true_labels)
        DCF[index] = norm_bayes_risk(M, pi_tilde, 1, 1)
        minDCF[index], _ = compute_min_DCF(llr, true_labels, pi_tilde, 1, 1)

    return DCF, minDCF


