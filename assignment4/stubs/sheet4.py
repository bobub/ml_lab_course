""" ps4_implementation.py

PUT YOUR NAME HERE:
Boris Bubla
Leonard Paeleke


Complete the classes and functions
- svm_qp
- plot_svm_2d
- neural_network
Write your implementations in the given functions stubs!


(c) Felix Brockherde, TU Berlin, 2013
    Jacob Kauffmann, TU Berlin, 2019
"""
import scipy.linalg as la
import matplotlib.pyplot as plt
import sklearn.svm
from cvxopt.solvers import qp, options
from cvxopt import matrix as cvxmatrix
import numpy as np
import torch as tr
from scipy.stats import bernoulli
import os
import scipy.io as sio
import random
from tqdm import tqdm
from copy import deepcopy
import itertools as it





class svm_qp():
    """ Support Vector Machines via Quadratic Programming """
    options['show_progress'] = False

    def __init__(self, kernel='linear', kernelparameter=1, C=1.):
        self.kernel = kernel
        self.kernelparameter = kernelparameter
        self.C = C
        self.alpha_sv = None
        self.b = None
        self.X_sv = None
        self.Y_sv = None

    def fit(self, X, Y):

        # INSERT_CODE

        self.X_sv = X
        self.Y_sv = Y
        self.__ydim = Y.shape[0]

        # reshape ytrain
        self.Y_sv = self.Y_sv.reshape(self.__ydim, -1)

        self.kernelmatrix = buildKernel(self.X_sv.T, kernel=self.kernel, kernelparameter=self.kernelparameter)

        P = (self.Y_sv @ self.Y_sv.T) * self.kernelmatrix
        q = -1 * np.ones((self.__ydim, 1))  # y_train has the same length as X_train
        A = self.Y_sv.T
        b = 0
        if self.C is None:
            # constraint for evrry every alpha: 0 =< alpha
            # use matrix notation
            # QP solver wants it: Gx =< h, where h expresses the 0 of the condition
            # to foollow the QP_solver formulation contraint in G expressed by -1
            G = -1 * np.identity(self.__ydim)
            h = np.zeros(self.__ydim)
        else:
            # constraint for every alpha: 0 =< alpha =< C
            # use matrix notation
            # QP solver wants it: Gx =< h, hence h expresses the upper and lower bound of the condition -> h_size = 2n x 1
            # G of size 2n x n
            # first n rows contain the constraint 0 =< alpha -> in G expressed by -1, in h with 0
            # second n rows contain the constraint alpha =< C -> in G expressed by 1, in h with C
            G = np.vstack((-1 * np.identity(self.__ydim), np.identity(self.__ydim)))
            h = np.hstack((np.zeros(self.__ydim), self.C * np.ones(self.__ydim)))
        # this is already implemented so you don't have to
        # read throught the cvxopt manual
        alpha = np.array(qp(cvxmatrix(P, tc='d'),
                            cvxmatrix(q, tc='d'),
                            cvxmatrix(G, tc='d'),
                            cvxmatrix(h, tc='d'),
                            cvxmatrix(A, tc='d'),
                            cvxmatrix(b, tc='d'))['x']).flatten()
        # Support vectors have non zero lagrange multipliers and are smaller than C/m
        self.sv = np.logical_and(alpha > 1e-5, alpha < np.round(self.C, 5))  # treshold 1e-5
        self.alpha_sv = alpha[self.sv]

        self.X_sv = self.X_sv[self.sv]
        self.Y_sv = self.Y_sv[self.sv]
        self.kernelmatrix = buildKernel(self.X_sv.T, kernel=self.kernel, kernelparameter=self.kernelparameter)

        # b = mean(y[sv] - sum(alpha*y*kernel(X_tr,X[sv]))
        # w is expressed by sum alpha*y*X_tr , usually it would be multiplied with X[sv].T for linear kernels
        # but because different kernels can be used, the kernel way is used

        # for all data points
        # self.b = np.mean(self.Y_sv-(alpha.reshape(-1,1)*self.__ytrain).T@self.kernelmatrix)

        # use only sv
        self.b = np.mean(self.Y_sv - (self.alpha_sv.reshape(-1, 1) * self.Y_sv).T @ (self.kernelmatrix))

    def predict(self, X):

        self.kernelmatrix = buildKernel(self.X_sv.T, X.T, kernel=self.kernel, kernelparameter=self.kernelparameter)
        self.yhat = np.sign((self.alpha_sv.reshape(-1, 1) * self.Y_sv).T @ self.kernelmatrix + self.b)

        return self.yhat

    def __linearKernel(self, Y):
        self.kernelmatrix = self.X_sv.dot(Y.T)

    def __polynomialKernel(self, Y):
        self.kernelmatrix = (self.X_sv.dot(Y.T) + 1) ** self.kernelparameter

    def __gaussianKernel(self, Y):
        X_len, X_width = self.X_sv.shape
        self.kernelmatrix = np.exp(-(
                np.diagonal(self.X_sv.dot(self.X_sv.T)).reshape(X_len, 1) - 2 * self.X_sv.dot(
            Y.T) + np.diagonal(Y.dot(Y.T))) / (2 * (self.kernelparameter ** 2)))


# This is already implemented for your convenience
class svm_sklearn():
    """ SVM via scikit-learn """
    def __init__(self, kernel='linear', kernelparameter=1., C=1.):
        if kernel == 'gaussian':
            kernel = 'rbf'
        self.clf = sklearn.svm.SVC(C=C,
                                   kernel=kernel,
                                   gamma=1./(1./2. * kernelparameter ** 2),
                                   degree=kernelparameter,
                                   coef0=kernelparameter)

    def fit(self, X, y):
        self.clf.fit(X, y)
        self.X_sv = X[self.clf.support_, :]
        self.y_sv = y[self.clf.support_]

    def predict(self, X):
        return self.clf.decision_function(X)


# Assignment 2
#this was only designed to handle y labels of shape (nx1) per function call (as on the instruction sheet).
# If you want to use it for multi class classification visualisation (as in test_nn_fit), loop through the labels.
def plot_boundary_2d(X, y, model):
    """
    Plots a 2 dimensional boundary of a model.

    Inputs:
    X = 2d data array (nx2)
    y = labels (nx1)
    model = model (typically SVM or neural net)
    """
    # 1. plot points X
    plt.scatter(X.T[0][np.argwhere(y == 1)], X.T[1][np.argwhere(y == 1)], c='b', label='Positive class')
    plt.scatter(X.T[0][np.argwhere(y == -1)], X.T[1][np.argwhere(y == -1)], c='r', label='Negative class')

    # 2. mark support vectors with a cross if svm
    if isinstance(model, svm_sklearn):
        plt.scatter(model.X_sv.T[0], model.X_sv.T[1], s=80, c='y', marker='x', label='Support vectors')

    # 3. plot separating hyperplane
    # 3a. create grid of predictions
    x_max = np.amax(X, axis=0)
    x_min = np.amin(X, axis=0)
    x0 = np.linspace(x_min[0], x_max[0], 50)
    x1 = np.linspace(x_min[1], x_max[1], 50)
    x0v, x1v = np.meshgrid(x0, x1)
    Xv = np.squeeze(np.array((x0v.reshape(2500, 1), x1v.reshape(2500, 1))))
    grid_pred = model.predict(Xv.T)
    # 3b plot level 0 contour line
    plt.contour(x0, x1, grid_pred.reshape(50, 50), levels=0)

    # format plot
    plt.ylabel('X1')
    plt.xlabel('X0')
    plt.title('2D visualisation of model classifications with a separating hyperplane')
    plt.legend()
    plt.show()


def sqdistmat(X, Y=False):
    if Y is False:
        X2 = sum(X**2, 0)[np.newaxis, :]
        D2 = X2 + X2.T - 2*np.dot(X.T, X)
    else:
        X2 = sum(X**2, 0)[:, np.newaxis]
        Y2 = sum(Y**2, 0)[np.newaxis, :]
        D2 = X2 + Y2 - 2*np.dot(X.T, Y)
    return D2


def buildKernel(X, Y=False, kernel='linear', kernelparameter=0):
    d, n = X.shape
    if Y is False:
        Y = X
    if kernel == 'linear':
        K = np.dot(X.T, Y)
    elif kernel == 'polynomial':
        K = np.dot(X.T, Y) + 1
        K = K**kernelparameter
    elif kernel == 'gaussian':
        K = sqdistmat(X, Y)
        K = np.exp(K / (-2 * kernelparameter**2))
    else:
        raise Exception('unspecified kernel')
    return K


# provided stub
class neural_network():
    def __init__(self, layers=[2, 100, 2], scale=.1, p=.1, lr=.1, lam=.1):
        super().__init__()
        self.weights = tr.nn.ParameterList([tr.nn.Parameter(scale * tr.randn(m, n)) for m, n in zip(layers[:-1], layers[1:])])
        self.biases = tr.nn.ParameterList([tr.nn.Parameter(scale * tr.randn(n)) for n in layers[1:]])
        self.parameters = list(self.weights) + list(self.biases)

        self.p = p
        self.lr = lr
        self.lam = lam
        self.train = False

    def relu(self, X, W, b):
        # algorithm 15, pg 46 from guide.pdf
        if self.train:
            delta = bernoulli.rvs(1 - self.p,
                                  size=W.shape[1])  # sample 'out' many samples from Bernoulli distribution B(1-p)
            Z = tr.from_numpy(delta) * tr.max(tr.zeros(X.shape[0], W.shape[1]), tr.mm(X, W) + b)

        else:
            Z = tr.max(tr.zeros(X.shape[0], W.shape[1]), (1 - self.p) * tr.mm(X, W) + b)

        return Z

    def softmax(self, Z, W, b):
        # algorithm 16, pg 46 from guide.pdf
        Z = tr.mm(Z, W) + b
        y_hat = tr.div(tr.exp(Z).T, tr.sum(tr.exp(Z), dim=1)).T

        return y_hat

    def forward(self, X):
        # algorithm 14, pg 45 from guide.pdf
        X = tr.tensor(X, dtype=tr.float)
        Z = X
        # apply ReLU to all layers but the last
        for w, b in zip(self.weights[:len(self.weights) - 1],
                        self.biases[:len(self.biases) - 1]):  # iterate through L-1 layers
            Z = self.relu(Z, w, b)
        # apply softmax to last layer
        y_hat = self.softmax(Z, self.weights[len(self.weights) - 1], self.biases[len(self.biases) - 1])

        return y_hat

    def predict(self, X):
        return self.forward(X).detach().numpy()

    def loss(self, ypred, ytrue):
        # compute cross entropy loss according to pg47 from guide.pdf
        loss = (-1 / ytrue.shape[0]) * tr.sum(ytrue * tr.log(ypred))

        return loss

    def fit(self, X, y, nsteps=1000, bs=100, plot=False):
        X, y = tr.tensor(X), tr.tensor(y)
        optimizer = tr.optim.SGD(self.parameters, lr=self.lr, weight_decay=self.lam)

        I = tr.randperm(X.shape[0])
        n = int(np.ceil(.1 * X.shape[0]))
        Xtrain, ytrain = X[I[:n]], y[I[:n]]
        Xval, yval = X[I[n:]], y[I[n:]]

        Ltrain, Lval, Aval = [], [], []
        for i in range(nsteps):
            optimizer.zero_grad()
            I = tr.randperm(Xtrain.shape[0])[:bs]
            self.train = True
            output = self.loss(self.forward(Xtrain[I]), ytrain[I])
            self.train = False
            Ltrain += [output.item()]
            output.backward()
            optimizer.step()

            outval = self.forward(Xval)
            Lval += [self.loss(outval, yval).item()]
            Aval += [np.array(outval.argmax(-1) == yval.argmax(-1)).mean()]

        if plot:
            plt.plot(range(nsteps), Ltrain, label='Training loss')
            plt.plot(range(nsteps), Lval, label='Validation loss')
            plt.plot(range(nsteps), Aval, label='Validation acc')
            plt.legend()
            plt.show()
            # END OF IMPLEMENTATION SECTION


"""


FUNCTIONS FROM ASSIGNMENT 3 AND OTHERS

These functions are used in the application assignments further below. 


"""

#cross entropy loss used in application assignment 6.2
def loss(ypred, ytrue):
    # compute cross entropy loss according to pg47 from guide.pdf
    loss = (-1/ytrue.shape[0])*np.sum(ytrue*np.log(ypred),axis=0)

    return loss
#from sheet3
def mean_absolute_error(y_true, y_pred):
    '''
    your code here
    '''
    loss = np.mean(abs(y_pred-y_true))
    return loss
#from sheet 3
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

#from sheet3
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
        # print(parameter[1])
        model = method(parameter[0], parameter[1], parameter[2])
        e = 0
        for i in range(nrepetitions):
            # Random Partitioning
            X_pos = np.linspace(0, length - 1, length)
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
        # print('Loss:' + str(model.cvloss))
        if e < method.cvloss:
            # print(e)
            method.cvloss = e
            # print('Loss in if:' + str(model.cvloss))
            method.__params = parameter

    return method

# adapted for neural network modelling, assignment 6.2
def cv_nn(X, y, method, params, loss_function=loss, nfolds=10, nrepetitions=5):
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
    method.cvloss = np.full((1, 10), 1000000)
    params_combinations = list(it.product(params['layers'], params['p'], params['lam'], params['lr']))
    for parameter in tqdm(params_combinations):
        # print(parameter[1])
        model = method(layers=parameter[0], p=parameter[1], lr=parameter[3], lam=parameter[2])
        e = np.zeros((1, 10))
        for i in range(nrepetitions):
            # Random Partitioning
            X_pos = np.linspace(0, length - 1, length)
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
                print('loss\n', loss_function(ypred=y_pred, ytrue=y[part[j].astype('int')]))
                e = e + loss_function(ypred=y_pred, ytrue=y[part[j].astype('int')])
        e = e / (nfolds * nrepetitions)
        # sum_e = e.sum()

        # print('Loss:' + str(model.cvloss))
        if e.sum() < method.cvloss.sum():
            # print(e)
            method.cvloss = e
            # print('Loss in if:' + str(model.cvloss))
            method.__params = parameter

    return method

"""
APPLICATION ASSIGNMENTS 4,5,6

These have been copied and adapted from a Jupyter notebook are not designed for reusability.
They are provided here as evidence of methodology and approach. If you would like the Jupyter notebook (which will be
much execution friendlier, just ask :) )

"""
class ASSIGNMENT4():
    def __init__(self):
        DATA_PATH = os.path.join(os.path.dirname(os.getcwd()), "data/easy_2d.npz")
        assert os.path.exists(DATA_PATH), "The path does not excist."
        data = np.load(DATA_PATH)
        print(data.files)
        self.X_tr = data['X_tr'].T
        self.Y_tr = data['Y_tr'].T
        self.X_te = data['X_te'].T
        self.Y_te = data['Y_te'].T
        plt.scatter(self.X_tr[self.Y_tr == -1, 0], self.X_tr[self.Y_tr == -1, 1], marker='x', color='r',
                    label='Negative -1')
        plt.scatter(self.X_tr[self.Y_tr == 1, 0], self.X_tr[self.Y_tr == 1, 1], marker='o', color='b',
                    label='Positive +1')

    def find_opti_parameters(self):
        params = {'kernel': ['gaussian'], 'kernelparameter': np.logspace(-5, 3, 50),
                  'regularization': np.logspace(-2, 3, 10)}
        self.cvsvm = cv(self.X_tr, self.Y_tr, svm_qp, params, loss_function=zero_one_loss, nfolds=5)
        self.params = self.cvsvm.params
        print("paramters found: {}".format(self.cvsvm.params))
        print("accuracy: {}".format(np.sum(np.sign(self.cvsvm.predict(self.X_te)) == self.Y_te) / len(self.X_te)))
        plot_boundary_2d(self.X_tr, self.Y_tr, self.cvsvm)

    def find_over_parameters(self):
        params = {'kernel': ['gaussian'], 'kernelparameter': np.logspace(-5, 3, 50),
                  'regularization': [10 ** 14]}  # np.logspace(13,16, 4) }
        self.cvsvm = cv(self.X_tr, self.Y_tr, svm_qp, params, loss_function=zero_one_loss, nfolds=5)
        print("paramters found: {}".format(self.cvsvm.params))
        plot_boundary_2d(self.X_tr, self.Y_tr, self.cvsvm)

    def find_under_parameters(self):
        params = {'kernel': ['gaussian'], 'kernelparameter': np.logspace(-7, -4, 50),
                  'regularization': [1]}  # np.logspace(1,10, 10) }
        self.cvsvm = cv(self.X_tr, self.Y_tr, svm_qp, params, loss_function=zero_one_loss, nfolds=5)
        print("paramters found: {}".format(self.cvsvm.params))
        plot_boundary_2d(self.X_tr, self.Y_tr, self.cvsvm)

    def ROC(self):
        print("find optimal parameters")
        self.cvsvm = self.find_opti_parameters()
        print("now create ROC curve")
        tpr = [np.zeros(1)]
        fpr = [np.zeros(1)]
        threshold = 0
        pos_label = -1
        neg = np.sum(np.array(self.Y_te.flatten() == -1))
        pos = len(self.Y_te) - neg
        # y_true=(y_true==pos_label) #boolean vec of true labels
        model = svm_qp(self.params[0], self.params[1], self.params[2])
        model.fit(self.X_tr, self.Y_tr)
        y_pred = model.predict(self.X_te)
        biases = y_pred
        # Use test predictions as bias
        # biases = np.linspace(-1,4,200)
        for bias in biases:
            prediction = y_pred.T + bias > threshold
            tpr.append(prediction[self.Y_te == 1].sum(axis=0) / pos)
            fpr.append(prediction[self.Y_te == -1, :].sum(axis=0) / neg)
        fpr = np.concatenate(fpr)
        tpr = np.concatenate(tpr)
        idx = np.argsort(fpr)  # sort by fpr in increasing order
        fpr = fpr[idx]
        tpr = tpr[idx]
        plt.plot(fpr, tpr, label='SVM')
        plt.plot(np.arange(0, 1.1, 0.1), np.arange(0, 1.1, 0.1), label='Random guesses')
        plt.ylabel('True Positive Rate (TPR)')
        plt.xlabel('False Positive Rate (FPR)')
        plt.title('ROC Curve')
        plt.legend()


class ASSIGNMENT5():
    def __init__(self):
        # Load data for #5
        DATA_PATH = os.path.join(os.path.dirname(os.getcwd()), "data/iris.npz")
        assert os.path.exists(DATA_PATH), "The path does not excist."
        data = np.load(DATA_PATH)
        print(data.files)
        self.X = data['X'].T
        self.Y = data['Y'].T
        print("Shape of X: {}".format(self.X.shape))
        print("Shape of Y: {}".format(self.Y.shape))
        # train test split
        test_size = 1 / 3
        idx = np.linspace(0, len(self.X) - 1, len(self.X)).astype(int)
        random.shuffle(idx)
        self.X_tr = self.X[idx[:int((1 - test_size) * len(self.X))], :]
        self.X_te = self.X[idx[int((1 - test_size) * len(self.X)):], :]
        self.y_tr = self.Y[idx[:int((1 - test_size) * len(self.X))]]
        self.y_te = self.Y[idx[int((1 - test_size) * len(self.X)):]]

    def visualize_data(self):
        fig, axs = plt.subplots(2, 2)
        axs[0, 0].scatter(self.X[self.Y == 1, 0], self.X[self.Y == 1, 1], marker='o', color='r', label='setosa')
        axs[0, 0].scatter(self.X[self.Y == 2, 0], self.X[self.Y == 2, 1], marker='o', color='b', label='versicolor')
        axs[0, 0].scatter(self.X[self.Y == 3, 0], self.X[self.Y == 3, 1], marker='o', color='g', label='virginica')
        axs[0, 1].scatter(self.X[self.Y == 1, 2], self.X[self.Y == 1, 1], marker='o', color='r', label='setosa')
        axs[0, 1].scatter(self.X[self.Y == 2, 2], self.X[self.Y == 2, 1], marker='o', color='b', label='versicolor')
        axs[0, 1].scatter(self.X[self.Y == 3, 2], self.X[self.Y == 3, 1], marker='o', color='g', label='virginica')
        axs[1, 0].scatter(self.X[self.Y == 1, 0], self.X[self.Y == 1, 3], marker='o', color='r', label='setosa')
        axs[1, 0].scatter(self.X[self.Y == 2, 0], self.X[self.Y == 2, 3], marker='o', color='b', label='versicolor')
        axs[1, 0].scatter(self.X[self.Y == 3, 0], self.X[self.Y == 3, 3], marker='o', color='g', label='virginica')
        axs[1, 1].scatter(self.X[self.Y == 1, 2], self.X[self.Y == 1, 3], marker='o', color='r', label='setosa')
        axs[1, 1].scatter(self.X[self.Y == 2, 2], self.X[self.Y == 2, 3], marker='o', color='b', label='versicolor')
        axs[1, 1].scatter(self.X[self.Y == 3, 2], self.X[self.Y == 3, 3], marker='o', color='g', label='virginica')

        axs[0, 0].set(ylabel='sepal width')
        axs[1, 0].set(xlabel='sepal length')
        axs[1, 0].set(ylabel='petal width')
        axs[1, 1].set(xlabel='petal length')

    def linear_hard_svm(self):
        accuracy = []
        for target in np.unique(self.Y):
            # relabel class data into 1 and -1
            # target is 1, rest -1
            y_tr_r = deepcopy(self.y_tr)
            y_tr_r[np.where(y_tr_r != target)] = -1
            y_tr_r[y_tr_r == target] = 1
            y_te_r = deepcopy(self.y_te)
            y_te_r[np.where(y_te_r != target)] = -1
            y_te_r[y_te_r == target] = 1

            # SVM with hard-margin
            params = {'kernel': ['linear'], 'kernelparameter': [1],
                      'regularization': [None]}  # np.logspace(-2,4, 200)*len(y_tr_r) }
            cvsvm = cv(self.X_tr, y_tr_r, svm_qp, params, loss_function=zero_one_loss, nfolds=5)
            print('Target class: {}'.format(target))
            print('Loss: {}'.format(cvsvm.cvloss))
            print('Parameters: {}'.format(cvsvm.params))
            accuracy.append(np.sum(np.sign(cvsvm.predict(self.X_te)) == y_te_r) / len(y_te_r))
        print('accuracy: {}'.format(accuracy))

    def gaussian_hard_svm(self):
        # gaussian kernel
        # hard margin
        accuracy = []
        for target in np.unique(self.Y):
            # relabel class data into 1 and -1
            # target is 1, rest -1
            y_tr_r = deepcopy(self.y_tr)
            y_tr_r[np.where(y_tr_r != target)] = -1
            y_tr_r[y_tr_r == target] = 1
            y_te_r = deepcopy(self.y_te)
            y_te_r[np.where(y_te_r != target)] = -1
            y_te_r[y_te_r == target] = 1

            # SVM with hard-margin
            params = {'kernel': ['gaussian'], 'kernelparameter': np.logspace(-2, 5, 100), 'regularization': [None]}
            cvsvm = cv(self.X_tr, y_tr_r, svm_qp, params, loss_function=zero_one_loss, nfolds=5)
            print('Target class: {}'.format(target))
            print('Loss: {}'.format(cvsvm.cvloss))
            print('Parameters: {}'.format(cvsvm.params))
            accuracy.append(np.sum(np.sign(cvsvm.predict(self.X_te)) == y_te_r) / len(y_te_r))
        print('accuracy: {}'.format(accuracy))

    def polynomial_hard_svm(self):
        # polynomial kernel
        # hard margin
        accuracy = []
        for target in np.unique(self.Y):
            # relabel class data into 1 and -1
            # target is 1, rest -1
            y_tr_r = deepcopy(self.y_tr)
            y_tr_r[np.where(y_tr_r != target)] = -1
            y_tr_r[y_tr_r == target] = 1
            y_te_r = deepcopy(self.y_te)
            y_te_r[np.where(y_te_r != target)] = -1
            y_te_r[y_te_r == target] = 1

            # SVM with hard-margin
            params = {'kernel': ['polynomial'], 'kernelparameter': np.logspace(-2, 1, 25), 'regularization': [None]}
            cvsvm = cv(self.X_tr, y_tr_r, svm_qp, params, loss_function=zero_one_loss, nfolds=5)
            print('Target class: {}'.format(target))
            print('Loss: {}'.format(cvsvm.cvloss))
            print('Parameters: {}'.format(cvsvm.params))
            accuracy.append(np.sum(np.sign(cvsvm.predict(self.X_te)) == y_te_r) / len(y_te_r))
        print('accuracy: {}'.format(accuracy))


def assignment_6_svm():
    # Assignment 6

    # cwd = os.getcwd()
    # print(cwd)
    # file_name = 'usps.mat'
    # path_to_data = cwd + file_name
    path_to_data = 'C:/Users/Boris/Desktop/ML Lab Course/ML Lab Assignments/assignment4/stubs/usps.mat'
    assert os.path.exists(path_to_data), "The path to the data does not exist."
    data = sio.loadmat(path_to_data)
    data_labels = data['data_labels']
    usps_data = data['data_patterns'].T

    X = usps_data
    y = data_labels

    # 6.1 Fivefold Cross validation - SVM

    # select 2005 random datapoints, so that we have equally sized folds
    X_pos = np.linspace(0, len(X) - 1, len(X))
    random.Random(2).shuffle(X_pos)
    X = X[X_pos[:2005].astype('int')]
    y = y.T[X_pos[:2005].astype('int')]
    print(X.shape)
    print(y.shape)

    # train test split so we can estimate test error
    from sklearn.model_selection import train_test_split
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

    # parameteres to search from
    params_lst = [{'kernel': ['gaussian'], 'kernelparameter': [0.1, 0.5, 0.9],
                   'regularization': [0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75]},
                  {'kernel': ['polynomial'], 'kernelparameter': [1, 2, 3],
                   'regularization': [0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75]},
                  {'kernel': ['linear'], 'kernelparameter': [0],
                   'regularization': [0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75]}]

    svm_classifiers = []
    # separate ylabels for one-vs-rest classification by looping through digit
    # for each digit
    for digit in y_tr.T:
        candidate_classifiers = []
        candidate_losses = []
        # for each parameter set
        for params in params_lst:
            # use cross entropy loss
            cvsvm = cv(X_tr, digit, svm_qp, params, loss_function=zero_one_loss, nfolds=4, nrepetitions=1)
            print(cvsvm)
            print(cvsvm.cvloss)
            print(cvsvm.__params)
            candidate_classifiers.append(cvsvm)
            candidate_losses.append(cvsvm.cvloss)

        # select classifier with lowest cv loss
        svm_classifiers.append(candidate_classifiers[np.argmin(candidate_losses)])

    results = [{'kernel': 'polynomial', 'kernelparameter': 2, 'C': 1.75},
               {'kernel': 'polynomial', 'kernelparameter': 2, 'C': 1.25},
               {'kernel': 'polynomial', 'kernelparameter': 2, 'C': 1},
               {'kernel': 'polynomial', 'kernelparameter': 2, 'C': [0.5]},
               {'kernel': 'polynomial', 'kernelparameter': 2, 'C': [1.75]},
               {'kernel': 'polynomial', 'kernelparameter': 2, 'C': [1.25]},
               {'kernel': 'polynomial', 'kernelparameter': [2], 'C': [1.5]},
               {'kernel': 'polynomial', 'kernelparameter': [2], 'C': [0.75]},
               {'kernel': 'polynomial', 'kernelparameter': [2], 'C': [0.5]},
               {'kernel': 'polynomial', 'kernelparameter': [2], 'C': [0.5]}, ]

    fig = plt.figure(figsize=(16, 18))
    fig.suptitle('Plot of 5 randomly chosen support vectors (columns) for digits 0-9 (rows)', fontsize=20)
    gs = fig.add_gridspec(10, 5)
    for idx, params in enumerate(results):
        # print(params['kernel'])
        svm = svm_qp(kernel=params['kernel'], kernelparameter=params['kernelparameter'], C=params['C'])
        svm.fit(X_tr, y_tr.T[idx])
        # print(svm.X_sv)
        # print(svm.X_sv.shape)

        # select 5 random SVs
        X_pos = np.linspace(0, len(svm.X_sv) - 1, len(svm.X_sv))
        random.Random(3).shuffle(X_pos)
        X = svm.X_sv[X_pos[:5].astype('int')]

        for idx2, sv in enumerate(X):
            f_ax1 = fig.add_subplot(gs[idx, idx2])
            plt.imshow(sv.reshape(16, 16), cmap='gray')


def assignment_6_neuralnet():
    # cwd = os.getcwd()
    # print(cwd)
    # file_name = 'usps.mat'
    # path_to_data = cwd + file_name
    path_to_data = 'C:/Users/Boris/Desktop/ML Lab Course/ML Lab Assignments/assignment4/stubs/usps.mat'
    assert os.path.exists(path_to_data), "The path to the data does not exist."
    data = sio.loadmat(path_to_data)
    data_labels = data['data_labels']
    usps_data = data['data_patterns'].T

    X = usps_data
    y = data_labels
    y = np.where(y == -1, 0, y)

    # select 2005 random datapoints, so that we have equally sized folds
    X_pos = np.linspace(0, len(X) - 1, len(X))
    random.Random(2).shuffle(X_pos)
    X = X[X_pos[:2005].astype('int')]
    y = y.T[X_pos[:2005].astype('int')]
    # print(X.shape)
    # print(y.shape)

    # parameters to search from
    params = {'layers': [[256, 20, 200, 10]], 'p': [0.05, 0.1, 0.15, 0.2], 'lam': [0.0001, 0.001, 0.01, 0.1],
              'lr': [0.01, 0.05, 0.1]}

    # try 'em out
    # cross entropy loss for classification
    cvnn = cv_nn(X, y, neural_network, params, loss_function=loss, nfolds=5, nrepetitions=1)

    nn = neural_network(cvnn.__params[0], scale=0.1, p=cvnn.__params[1], lr=cvnn.__params[3], lam=cvnn.__params[2])
    nn.fit(X, y)

    fig = plt.figure(figsize=(12, 16))
    gs = fig.add_gridspec(5, 4)
    fig.suptitle('Plots of 20 weight vectors from first layer of neural net', fontsize=20)

    for idx, weight in enumerate(nn.weights[0].T):
        # print(weight.shape)
        f_ax1 = fig.add_subplot(gs[idx - 1])
        plt.title('Weight vector %d' % idx)
        plt.imshow(weight.detach().numpy().reshape(16, 16), cmap='gray')

