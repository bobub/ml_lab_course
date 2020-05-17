""" sheet1_implementation.py

PUT YOUR NAME HERE:
Boris Bubla
Leonard Paeleke


Write the functions
- pca
- gammaidx
- lle
Write your implementations in the given functions stubs!


(c) Daniel Bartz, TU Berlin, 2013
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse.csgraph import connected_components
import os
import scipy.linalg as la
import scipy.spatial as sp
from scipy.linalg import expm
import matplotlib.gridspec as gridspec
import scipy.io as sio


class PCA():
    """
        Definition of PCA Class
        Algorithms in guide.pdf used, pg 15-17
    """

    def __init__(self, Xtrain):
        """
            Compute Principal Components
        """
        # 1. centre data
        self.Xmean = np.mean(Xtrain, axis=0)
        self.C = Xtrain - self.Xmean
        # 2. generate covariance marix
        self.C = np.cov(self.C, rowvar=False)
        # 3. calculate eigenvalues and eigenvectors
        self.D, self.U = np.linalg.eigh(self.C)

        self.idx = np.argsort(self.D)[::-1]
        # Sort the eigenvalue from high to low
        self.D = self.D[self.idx]
        # sort eigenvectors according to same index
        self.U = self.U[:, self.idx]

    def project(self, Xtest, m):
        """
            Projecting to the low-dimensional sub-space
        """
        # 1. centre data by mean of training
        Xtest = Xtest - self.Xmean
        # 2. project data to m principal components
        Z = self.U.T[range(m)].dot(Xtest.T).T
        return Z

    def denoise(self, Xtest, m):
        """
            Reconstructing projected data points in the original space
        """
        # 1. projection to the low-dimencsional sub-space
        Z = self.project(Xtest, m)
        # 2. recontruction by m dimensions
        Y = Z.dot(self.U.T[range(m)]) + self.Xmean
        return Y


def gammaidx(X, k):
    """
    Gamma identification for outlier detection by ranking
    """
    y = []
    # Calculate distance matrix
    D = np.linalg.norm(X[None, :] - X[:, None], axis=2)
    # Sort distance matrix
    kn = np.argsort(D, kind='mergesort')
    # identify k-nearest neighbours
    kn = kn[:, 1:k + 1]
    # sum over k-neaest neighbours and divide bei k
    y = np.sum(np.take_along_axis(D, kn, axis=1), axis=1) / k

    return y


def auc(y_true, y_pred, plot=False):
    # 1. FIND ROC CURVE POINTS & FPR/TPR
    pos_label = 1
    y_true = (y_true == pos_label)  # boolean vec of true labels

    # arrange predictions in descending order (indexes)
    descending_scores = np.argsort(y_pred, kind='mergesort')[::-1]
    # ascending_scores=np.argsort(y_pred,kind='mergesort')[::1]
    y_pred = y_pred[descending_scores]
    y_true = y_true[descending_scores]

    # determine distinct values to create an index of decreasing values
    # 'predicted value in y_pred where lower values tend to correspond to label -1 and higher values to label +1'
    distinct_values_idx = np.where(np.diff(y_pred))[0]  # length n-1 as calculating differences
    distinct_descending_scores_idx = np.r_[distinct_values_idx, y_true.size - 1]  # add last entry

    tps = np.cumsum(y_true)[distinct_descending_scores_idx]  # cumulative sum of true positives using idx
    fps = 1 - tps + distinct_descending_scores_idx  # same as cum sum of false positives

    # add 0,0 position for ROC curve
    tps = np.r_[0, tps]
    fps = np.r_[0, fps]

    # false/true positive rate
    fpr = fps / fps[-1]  # rate=sum/max
    tpr = tps / tps[-1]

    # 2.PLOT ROC CURVE POINTS
    if plot == True:
        plt.plot(fpr, tpr, label='Algorithm')
        plt.plot(np.arange(0, 1.1, 0.1), np.arange(0, 1.1, 0.1), label='Random guesses')
        plt.ylabel('True Positive Rate (TPR)')
        plt.xlabel('False Positive Rate (FPR)')
        plt.title('ROC Curve')
        plt.legend()

    # 3. CALCULATE AUC
    # reshape needed
    fpr = fpr.reshape(1, fpr.shape[0])
    tpr = tpr.reshape(1, tpr.shape[0])

    # assume positive area
    end = 1
    # check if negative area (good discrimination, just switch labels)
    diff_fpr = np.diff(fpr)
    if np.all(diff_fpr <= 0):
        end = -1
    # calculate area using trapezoidal approach
    area = end * np.trapz(tpr, fpr)

    return area


def lle(X, m, tol, n_rule, k=None, epsilon=None):
    """
        Locally Linear Embedding
    """

    # compute neighborhoord by kNN or eps-bole rule

    # 1. calculate euclidean distance of data
    D = np.sqrt(np.sum((X[None, :] - X[:, None]) ** 2, -1))

    # 2. check for applied rule
    if n_rule == 'knn':
        # check if k is provided
        assert (k != None), """The parameter 'k' is required for the 'knn' rule"""
        # 3a. calculate k nearest neighbors
        # Sort distance matrix
        kn = np.argsort(D, kind='mergesort')
        # identify k-nearest neighbors
        kn = kn[:, 1:k + 1]

    elif n_rule == 'eps-ball':
        # check if epsilon is provided
        assert (epsilon != None), """The parameter 'epsilon' is required for the 'eps-ball' rule"""
        # 3b. compare distance by epsilon
        # tupel (1. element, 2. element)
        # idx = np.argwhere(D<epsilon)
        # row wise all points in the neighborhood
        # kn = [idx[:,1][idx[:,0]==i] for i in np.unique(idx[:,0])]

        # 3b. compare distance by epsilon
        # boolean approach
        kn = D < epsilon
    else:
        print("""The following rule {} is not known. Please use either 'knn' or 'eps-ball'.""".format(n_rule))

    # 4. calculate reconstruction weights
    # intialize weight matrix
    W = np.zeros((len(X), len(X)))
    # calculate weights for every point
    for i in range(len(X)):
        # calculate covariance matrix
        C = np.cov(X[i] - X[kn[i]])
        # solve for weights
        I = np.eye(len(X[kn[i]]))
        weights = np.linalg.inv(C - tol * I).dot(np.ones(len(X[kn[i]])).reshape(len(X[kn[i]]), 1))
        # normalize weights
        weights = (1 / (weights.T.dot(np.ones(len(X[kn[i]])).reshape(len(X[kn[i]]), 1))) * weights).reshape(
            len(X[kn[i]]))
        W[i, kn[i]] = weights

    # 5. calculate cost matrix
    I = np.eye(len(X))
    M = (I - W).T @ (I - W)

    # 6. Obtain eigenvalues and eigenvector of M
    eigen_values, eigen_vector = np.linalg.eigh(M)
    # sort eigenvalues in ascending order
    eigen_kn = np.argsort(abs(eigen_values), kind='mergesort').reshape(len(X), 1)
    # sort eigenvectors by eigenvalues, eigenvector along columns, first eigenvector -> [:,0]
    V = -1 * np.take_along_axis(eigen_vector.T, eigen_kn,
                                axis=0)  # CAVEAT: mulitplying by -1 because numpy.eig routine delivers wrong sign

    # 7. embedded dimension
    Y = V[:, 1:m + 1]

    # 8 Check connected graphs
    if (connected_components(V, directed=False)[0] != 1):
        raise ValueError('Graphs are not connected!')

    return Y