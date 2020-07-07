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
import random
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from scipy.stats import lognorm
from tqdm import tqdm


def zero_one_loss(y_true, y_pred):
    '''
    Applies a the zero one loss function to predictions.

    Input:
    y_true = the true data labels (nx1)
    y_pred = the predicted data labels (nx1)

    Output:
    loss = the zero one loss
    '''
    loss = np.count_nonzero(y_true != np.sign(y_pred))
    return loss


def mean_absolute_error(y_true, y_pred):
    '''
    your code here
    '''
    loss = np.mean(abs(y_pred-y_true))
    return loss


def cv(X, y, method, params, loss_function=mean_absolute_error, nfolds=10, nrepetitions=5):
    '''
    This function applies a cross validation procedure with a specified loss function and model parameters.

    Inputs:
    X = X training data (nxd)
    y = y training data (nxe)
    method = model
    params = the model parameters: the 'kernel', the 'kernelparameter' and 'regularisation' term.
    loss_function = the loss function to be used to calculate error.
    nfolds = number of equal sized folds
    nrepetitions = number of repetitions

    Outputs:
    method = the model object with the best parameters and cvloss.
    '''
    length, width = np.shape(X)
    # model = method
    method.cvloss = 1000000
    params_combinations = list(it.product(params['kernel'], params['kernelparameter'], params['regularization']))
    for parameter in tqdm(params_combinations):
        #print(parameter[1])
        model = method(parameter[0], parameter[1], parameter[2])
        e = 0
        for i in range(nrepetitions):
            # Random Partitioning
            X_pos = np.linspace(0,length-1, length)
            random.shuffle(X_pos)
            part = np.array_split(X_pos, nfolds)
            for j in range(nfolds):
                # Assign every part not j as training set
                # Xtr indices
                train = np.concatenate(np.array(part)[tuple([np.array(range(nfolds)) != j])].astype('int'))
                X_j = X[train]
                y_j = y[train]
                model.fit(X_j, y_j)
                y_pred = model.predict(X[part[j].astype('int')])
                e = e + loss_function(y[part[j].astype('int')], y_pred)
        e = e / (nfolds * nrepetitions)
        #print('Loss:' + str(model.cvloss))
        if e < method.cvloss:
            #print(e)
            method.cvloss = e
            #print('Loss in if:' + str(model.cvloss))
            method.__params = parameter
    #print(method.params)
    method = model.fit(X,y,method.__params[0],method.__params[1],method.__params[2])
    return method


class krr():
    '''
    This class is used for kernel ridge regression.

    Attributes:
        self.kernel = kernel to be used. Can be linear, gaussian, polynomial.
        self.kernelparameter = the value of the kernelparameter, if relevant.
        self.regularisation = the regularisation constant. If 0, efficient LOOCV is used.

    Methods:
        fit = fits given data to krr model with given parameters.
        pred = uses fitted model to predict new unseen test data.
        __linearKernel = implements a linear kernel
        __polynomialKernel = implements a polynomial kernel with kernelparameter
        __gaussianKernel = implements a gaussian kernel with kernelparameter.
        LOOCV = implements efficient leave one out cross validation to find a regularisation constant for the model.

    Outputs:
    the krr object
    '''

    def __init__(self, kernel='linear', kernelparameter=1, regularization=0):
        self.kernel = kernel
        self.kernelparameter = kernelparameter
        self.regularization = regularization

    def fit(self, X, y, kernel=False, kernelparameter=False, regularization=False):
        '''
        Fits training data to the specified model parameters by calculating alpha. Uses LOOCV is regularization =0.

        Inputs:
        X = training data X
        y = training data y
        kernel = kernel (see krr attributes)
        kernelparameter = kernelparameter
        regularisation = regularisation term

        Outputs:
        fitted model with optimised alpha.

        '''
        self.__Xtrain = X
        self.__ytrain = y
        self.__ydim = y.shape[1]

        if kernel is not False:
            self.kernel = kernel
        if kernelparameter is not False:
            self.kernelparameter = kernelparameter
        if regularization is not False:
            self.regularization = regularization
        # calculate kernelmatrix
        if self.kernel == 'linear':
            self.__linearKernel(X)
        elif self.kernel == 'polynomial':
            self.__polynomialKernel(X)
        elif self.kernel == 'gaussian':
            self.__gaussianKernel(X)
        else:
            print(
                """The following kernel {} is not known. Please use either 'linear' , 'polynomial' or 'gaussian'.""".format(
                    kernel))
        if self.regularization == 0:
            self.__LOOCV()

        # calculate optimized alpha
        I_length = len(self.kernelmatrix)
        self.alpha = np.linalg.solve(self.kernelmatrix + self.regularization * np.identity(I_length),
                                     self.__ytrain).reshape(-1, len(self.__ytrain))

        return self

    def predict(self, X):
        '''
        Makes y-predictions based on x testing data.

        Input:
        X = xtesting data

        Output
        y_pred = y predictions

        '''
        # calculate kernelmatrix
        if self.kernel == 'linear':
            self.__linearKernel(X)
        elif self.kernel == 'polynomial':
            self.__polynomialKernel(X)
        elif self.kernel == 'gaussian':
            self.__gaussianKernel(X)
        # calculate prediction
        y_pred = self.alpha.dot(self.kernelmatrix)  # <alpha,kernelmatrix>
        return y_pred.reshape(len(X), self.__ydim)  # len(self.alpha)

    def __linearKernel(self, Y):
        self.kernelmatrix = self.__Xtrain.dot(Y.T)

    def __polynomialKernel(self, Y):
        self.kernelmatrix = (self.__Xtrain.dot(Y.T) + 1) ** self.kernelparameter

    def __gaussianKernel(self, Y):
        X_len, X_width = self.__Xtrain.shape
        self.kernelmatrix = np.exp(-(
                    np.diagonal(self.__Xtrain.dot(self.__Xtrain.T)).reshape(X_len, 1) - 2 * self.__Xtrain.dot(
                Y.T) + np.diagonal(Y.dot(Y.T))) / (2 * (self.kernelparameter ** 2)))

    def __LOOCV(self):
        """
        Finds the regularisation constant according to efficient leave one out cross validation as in guide.pdf
        """
        # Leave-One-Out-Cross-Validation
        # Eigenvalue decomposition
        squared_loss = []
        L, U = np.linalg.eigh(self.kernelmatrix)  # L = Eigenvalue, U = Eigenvector
        mean_L = np.mean(L)
        I = np.identity(len(L))
        # for faster computation precalculate U.T.y
        UTy = U.T.dot(self.__ytrain)
        # logarithmic distribution with mu = mean_L and sigma = 1
        # create 50 values of C
        # identify C around Kernel eigenvalue means with logarithmic distribution
        Cs = np.logspace(-10, 10, 50) * mean_L
        for C in Cs:
            ULCI = U.dot(L * I).dot((1 / (L + C)) * I)  # (1/(L + C))*I: inverse of diagonal matrix
            squared_loss.append(
                np.sum(((self.__ytrain - ULCI.dot(UTy)) / (1 - np.diagonal(ULCI.dot(U.T)))) ** 2) / len(self.__ytrain))
        self.regularization = Cs[np.argmin(squared_loss)]

        return self


def roc_fun(y_true, y_pred, biases=100, threshold=0):
    """
    ROC function used to plot average ROC curve for assignment 4c.

    It creates a set of predictions for each bias and then average over biases to get an average TPR and FPR.

    This is then used a loss function in CV.
    """
    y_pred = y_pred[:, np.newaxis]

    b = np.linspace(-1, 1, biases)
    prediction = (y_pred - b) > threshold

    neg = np.sum(np.array(y_true.flatten() == -1))
    pos = len(y_true) - neg

    tpr = prediction[y_true.flatten() == 1, :].sum(axis=0) / pos
    # print('tpr\n',np.shape(tpr))
    fpr = prediction[y_true.flatten() == -1, :].sum(axis=0) / neg
    # print(fpr)
    result = np.squeeze(np.array([fpr, tpr]))

    return result





