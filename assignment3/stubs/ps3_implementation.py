""" ps3_implementation.py

PUT YOUR NAME HERE:
<FIRST_NAME><LAST_NAME>


Write the functions
- cv
- zero_one_loss
- krr
Write your implementations in the given functions stubs!


(c) Daniel Bartz, TU Berlin, 2013
"""
import numpy as np
import scipy.linalg as la
import itertools as it
import time
import pylab as pl
from mpl_toolkits.mplot3d import Axes3D


def zero_one_loss(y_true, y_pred):
    ''' your header here!
    '''


def mean_absolute_error(y_true, y_pred):
    ''' your code here '''


def cv(X, y, method, params, loss_function=zero_one_loss, nfolds=10, nrepetitions=5):
    ''' your header here!
    '''
    return method


class krr():
    """
    Class used to apply kernel ridge regression.

    Inputs:
    kernel = kernel to be used (linear, polynomial, gaussian)
    kernelparameter = none, degree d, kernel width sigma (for linear, polynomial, gaussian respectively)
    regularization = regularisation constant C (default is 0, which then uses leave one out cross validation LOOCV as defined)

    Methods:
    fit: fits Xtrain and ytrain with the desired kernel
    predict: makes predictions using fit on new Xtest data
    """

    def __init__(self,kernel='linear',kernelparameter=1,regularization=0,alpha=None,K_matrix=None,Xtrain=None,ytrain=None):
        self.kernel = kernel
        self.alpha = alpha
        self.kernelparameter = kernelparameter
        self.regularization = regularization
        self.K_matrix = K_matrix
        self.Xtrain = Xtrain
        self.ytrain = ytrain

    def K_linear(self, x, x_):
        k = np.dot(x, x_.T)  # or np.inner()?
        return k

    def K_poly(self, x, x_):
        k = (np.dot(x, x_.T) + 1) ** self.kernelparameter
        return k

    def K_gaussian(self, x, x_):
        k = np.exp(-(np.linalg.norm(x - x_.T) ** 2) / (2 * (self.kernelparameter) ** 2))
        return k

    def LOOCV(self, K_matrix, num=20, power=2):
        """
        Executes leave one out cross validation for kernel ridge regression for automatic selection of C, the regularisation parameter.
        Uses mean eigenvalue as centre of logarithmically spaced candidates for C.

        Inputs:
        self = KRR instance
        K_matrix = gram matrix
        num = number of candidates for C
        power = max power of logarithmic scale above/below mean eigenvalue.

        Output:
        best_C = the best regularisation constant which gives the lowest cross validation error, epsilon

        """

        # use EVD's results for efficient computation
        eigval, eigvec = np.linalg.eig(K_matrix)

        # create candidates
        a = np.logspace(-power, power, num)
        C = a * np.mean(eigval)  # candidates denoted as c

        # compute S
        # cxnxn
        S = (np.dot(eigvec, np.diag(eigval))[None, :, :] * np.linalg.inv(
            np.diag(eigval) + C[:, None, None] * np.eye(K_matrix.shape[0])[None, :, :])) * eigvec.T[None, :, :]

        # cxnx1 = cxnxn X nx1
        Sy = np.dot(S, self.ytrain)

        # calculate all quadratic losses for c candidates
        # c   =    0xnx1-cxnx1 / (1-cxn)
        epsilon = np.sum((((self.ytrain[None, :, :] - Sy) / (1 - np.diagonal(S, axis1=1, axis2=2))[:, :, None]) ** 2),
                         axis=1)

        # best_epsilon = epsilon[np.argmin(epsilon)]

        best_C = C[np.argmin(epsilon)]

        return best_C

    def fit(self, Xtrain, ytrain):

        self.Xtrain = Xtrain
        self.ytrain = ytrain

        if self.kernel == 'linear':
            self.K_matrix = krr.K_linear(Xtrain, Xtrain)

        if self.kernel == 'polynomial':
            self.K_matrix = krr.K_poly(Xtrain, Xtrain)

        if self.kernel == 'gaussian':
            self.K_matrix = krr.K_gaussian(Xtrain, Xtrain)

        if self.regularization == 0:
            self.regularization = krr.LOOCV(self.K_matrix)

        # nx1 =                  (nxn)^-1            x         nx1            = nx1
        self.alpha = np.dot(np.inv(self.K_matrix + self.regularization * np.eye(Xtrain.shape[0])), ytrain)

    def predict(self, Xtest):
        # nx1 = nx1 X
        if self.kernel == 'linear':
            y_pred = np.inner(self.alpha.T, krr.K_linear(self.Xtrain, Xtest).T)

        if self.kernel == 'polynomial':
            y_pred = np.inner(self.alpha.T, krr.K_poly(self.Xtrain, Xtest).T)

        if self.kernel == 'gaussian':
            y_pred = np.inner(self.alpha.T, krr.K_gaussian(self.Xtrain, Xtest).T)

        return y_pred






