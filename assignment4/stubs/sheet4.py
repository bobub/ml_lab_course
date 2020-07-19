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
import itertools as it
import time
import pylab as pl
import random
from scipy.stats import lognorm
from tqdm import tqdm
from copy import deepcopy



class svm_qp():
    """ Support Vector Machines via Quadratic Programming """
    options['show_progress'] = False
    
    def __init__(self, kernel='linear', kernelparameter=1., C=1.):
        self.kernel = kernel
        self.kernelparameter = kernelparameter
        self.C = C
        self.alpha_sv = None
        self.b = None
        self.X_sv = None
        self.Y_sv = None
        
    
    def fit(self, X, Y):

        # INSERT_CODE
        
        # Here you have to set the matrices as in the general QP problem
        #P = 
        #q = 
        #G = 
        #h = 
        #A =   # hint: this has to be a row vector
        #b =   # hint: this has to be a scalar
        
        self.X_sv = X
        self.Y_sv = Y
        self.__ydim = Y.shape[0]
        
        # reshape ytrain
        self.Y_sv = self.Y_sv.reshape(self.__ydim,-1)

        

        self.kernelmatrix = buildKernel(self.X_sv.T, kernel = self.kernel, kernelparameter = self.kernelparameter)
        
        P = (self.Y_sv@self.Y_sv.T)*self.kernelmatrix
        q = -1*np.ones((self.__ydim,1)) #y_train has the same length as X_train
        A = self.Y_sv.T
        b = 0
        if self.C is None: 
            # no regularization C yields hard-margin
            # constraint for evrry every alpha: 0 =< alpha
            # use matrix notation
            # QP solver wants it: Gx =< h, where h expresses the 0 of the condition
            # to foollow the QP_solver formulation contraint in G expressed by -1
            G = -1*np.identity(self.__ydim)
            h = np.zeros(self.__ydim)
        else:
            # constraint for every alpha: 0 =< alpha =< C
            # use matrix notation
            # QP solver wants it: Gx =< h, hence h expresses the upper and lower bound of the condition -> h_size = 2n x 1
            # G of size 2n x n 
            # first n rows contain the constraint 0 =< alpha -> in G expressed by -1, in h with 0
            # second n rows contain the constraint alpha =< C -> in G expressed by 1, in h with C
            G = np.vstack((-1*np.identity(self.__ydim),np.identity(self.__ydim)))
            h = np.hstack((np.zeros(self.__ydim),self.C/self.__ydim*np.ones(self.__ydim)))
        # this is already implemented so you don't have to
        # read throught the cvxopt manual
        alpha = np.array(qp(cvxmatrix(P, tc='d'),
                            cvxmatrix(q, tc='d'),
                            cvxmatrix(G, tc='d'),
                            cvxmatrix(h, tc='d'),
                            cvxmatrix(A, tc='d'),
                            cvxmatrix(b, tc='d'))['x']).flatten()
        # Support vectors have non zero lagrange multipliers and are smaller than C/m
        if self.C is not None:
            self.sv = np.logical_and(alpha > 1e-5, alpha < np.round(self.C/self.__ydim,5)) #treshold 1e-5
        else: 
            # no regularization C yields hard-margin
            self.sv = alpha > 1e-5
        self.alpha_sv = alpha[self.sv]
        
        self.X_sv = self.X_sv[self.sv]
        self.Y_sv = self.Y_sv[self.sv]
    
        # calculation of bias b
    
        self.kernelmatrix = buildKernel(self.X_sv.T, kernel = self.kernel, kernelparameter = self.kernelparameter)
        
        # b = mean(y[sv] - sum(alpha*y*kernel(X_tr,X[sv]))
        # w is expressed by sum alpha*y*X_tr , usually it would be multiplied with X[sv].T for linear kernels
        # but because different kernels can be used, the kernel way is used 
        
        # for all data points
        #self.b = np.mean(self.Y_sv-(alpha.reshape(-1,1)*self.__ytrain).T@self.kernelmatrix)
        
        # use only sv 
        self.b = np.mean(self.Y_sv-(self.alpha_sv.reshape(-1,1)*self.Y_sv).T@(self.kernelmatrix))
        
    def predict(self, X):

        # INSERT_CODE

        self.kernelmatrix = buildKernel(self.X_sv.T,X.T, kernel = self.kernel, kernelparameter = self.kernelparameter)
        self.yhat = (self.alpha_sv.reshape(-1,1)*self.Y_sv).T @ self.kernelmatrix + self.b

        return self.yhat



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
    if isinstance(model, svm_qp):
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
    if isinstance(Y,bool) and Y is False:
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

def mean_absolute_error(y_true, y_pred):
    ''' 
    your code here 
    '''
    loss = np.mean(abs(y_pred-y_true))
    #loss = np.sum(np.sum((y_pred-y_true)**2)**0.5, axis = 1) / len(y_pred)
    return loss

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

def cv(X, y, method, params, loss_function=mean_absolute_error, nfolds=10, nrepetitions=5):
    ''' 
    your header here!
    '''
    # TODO progress bar, run time estimation
    length, width = np.shape(X)
    # model = method
    method.cvloss = 1000000
    method.params = None
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
                X_j = X[train,:]
                y_j = y[train]
                model.fit(X_j, y_j)
                y_pred = np.sign(model.predict(X[part[j].astype('int')]))
                e = e + loss_function(y[part[j].astype('int')], y_pred)
        e = e / (nfolds * nrepetitions)
        #print('Loss:' + str(e))
        if e < method.cvloss:
            print(e)
            method.cvloss = e
            #print('Loss in if:' + str(model.cvloss))
            method.params = parameter
    #print(method.params)
    model = method(method.params[0],method.params[1],method.params[2])
    model.fit(X,y)
    method = model
    return method

class ASSIGNMENT4():
    def __init__(self):
        DATA_PATH = os.path.join(os.path.dirname(os.getcwd()),"data/easy_2d.npz")
        assert os.path.exists(DATA_PATH), "The path does not excist."
        data = np.load(DATA_PATH)
        print(data.files)
        self.X_tr = data['X_tr'].T
        self.Y_tr = data['Y_tr'].T
        self.X_te = data['X_te'].T
        self.Y_te = data['Y_te'].T
        plt.scatter(self.X_tr[self.Y_tr == -1,0], self.X_tr[self.Y_tr == -1,1], marker = 'x', color = 'r', label = 'Negative -1')
        plt.scatter(self.X_tr[self.Y_tr == 1,0], self.X_tr[self.Y_tr == 1,1], marker = 'o', color = 'b',label = 'Positive +1')
    def find_opti_parameters(self):
        params = { 'kernel': ['gaussian'], 'kernelparameter': np.logspace(-5,3, 50), 'regularization': np.logspace(-2,3, 10) }
        self.cvsvm = cv(self.X_tr, self.Y_tr, svm_qp, params, loss_function=zero_one_loss, nfolds=5)
        self.params = self.cvsvm.params
        print("paramters found: {}".format(self.cvsvm.params))
        print("accuracy: {}".format(np.sum(np.sign(self.cvsvm.predict(self.X_te)) == self.Y_te)/ len(self.X_te)))
        plot_boundary_2d(self.X_tr, self.Y_tr, self.cvsvm)
    def find_over_parameters(self):
        params = { 'kernel': ['gaussian'], 'kernelparameter': np.logspace(-5,3, 50), 'regularization': [10**14]}#np.logspace(13,16, 4) }
        self.cvsvm = cv(self.X_tr, self.Y_tr, svm_qp, params, loss_function=zero_one_loss, nfolds=5)
        print("paramters found: {}".format(self.cvsvm.params))
        plot_boundary_2d(self.X_tr, self.Y_tr, self.cvsvm)
    def find_under_parameters(self):
        params = { 'kernel': ['gaussian'], 'kernelparameter': np.logspace(-7,-4, 50), 'regularization': [1]}#np.logspace(1,10, 10) }
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
        pos_label=-1
        neg = np.sum(np.array(self.Y_te.flatten() == -1))
        pos = len(self.Y_te) - neg
        #y_true=(y_true==pos_label) #boolean vec of true labels
        model = svm_qp(self.params[0],self.params[1],self.params[2])
        model.fit(self.X_tr,self.Y_tr)
        y_pred = model.predict(self.X_te)
        biases = y_pred
        # Use test predictions as bias 
        #biases = np.linspace(-1,4,200)
        for bias in biases:
            prediction = y_pred.T + bias > threshold
            tpr.append(prediction[self.Y_te == 1].sum(axis=0) / pos)
            fpr.append(prediction[self.Y_te == -1, :].sum(axis=0) / neg)
        fpr = np.concatenate(fpr)
        tpr = np.concatenate(tpr)
        idx = np.argsort(fpr) # sort by fpr in increasing order
        fpr = fpr[idx]
        tpr = tpr[idx]
        plt.plot(fpr,tpr,label='SVM')
        plt.plot(np.arange(0,1.1,0.1),np.arange(0,1.1,0.1),label='Random guesses')
        plt.ylabel('True Positive Rate (TPR)')
        plt.xlabel('False Positive Rate (FPR)')
        plt.title('ROC Curve')
        plt.legend()
class ASSIGNMENT5():
    def __init__(self):
        # Load data for #5
        DATA_PATH = os.path.join(os.path.dirname(os.getcwd()),"data/iris.npz")
        assert os.path.exists(DATA_PATH), "The path does not excist."
        data = np.load(DATA_PATH)
        print(data.files)
        self.X = data['X'].T
        self.Y = data['Y'].T
        print("Shape of X: {}".format(self.X.shape))
        print("Shape of Y: {}".format(self.Y.shape))
        # train test split
        test_size = 1/3
        idx = np.linspace(0,len(self.X)-1,len(self.X)).astype(int)
        random.shuffle(idx)
        self.X_tr = self.X[idx[:int((1-test_size)*len(self.X))],:]
        self.X_te = self.X[idx[int((1-test_size)*len(self.X)):],:]
        self.y_tr = self.Y[idx[:int((1-test_size)*len(self.X))]]
        self.y_te = self.Y[idx[int((1-test_size)*len(self.X)):]]
    def visualize_data(self):
        fig, axs = plt.subplots(2, 2)
        axs[0, 0].scatter(self.X[self.Y == 1,0], self.X[self.Y == 1,1], marker = 'o', color = 'r', label = 'setosa')
        axs[0, 0].scatter(self.X[self.Y == 2,0], self.X[self.Y == 2,1], marker = 'o', color = 'b',label = 'versicolor')
        axs[0, 0].scatter(self.X[self.Y == 3,0], self.X[self.Y == 3,1], marker = 'o', color = 'g',label = 'virginica')
        axs[0, 1].scatter(self.X[self.Y == 1,2], self.X[self.Y == 1,1], marker = 'o', color = 'r', label = 'setosa')
        axs[0, 1].scatter(self.X[self.Y == 2,2], self.X[self.Y == 2,1], marker = 'o', color = 'b',label = 'versicolor')
        axs[0, 1].scatter(self.X[self.Y == 3,2], self.X[self.Y == 3,1], marker = 'o', color = 'g',label = 'virginica')
        axs[1, 0].scatter(self.X[self.Y == 1,0], self.X[self.Y == 1,3], marker = 'o', color = 'r', label = 'setosa')
        axs[1, 0].scatter(self.X[self.Y == 2,0], self.X[self.Y == 2,3], marker = 'o', color = 'b',label = 'versicolor')
        axs[1, 0].scatter(self.X[self.Y == 3,0], self.X[self.Y == 3,3], marker = 'o', color = 'g',label = 'virginica')
        axs[1, 1].scatter(self.X[self.Y == 1,2], self.X[self.Y == 1,3], marker = 'o', color = 'r', label = 'setosa')
        axs[1, 1].scatter(self.X[self.Y == 2,2], self.X[self.Y == 2,3], marker = 'o', color = 'b',label = 'versicolor')
        axs[1, 1].scatter(self.X[self.Y == 3,2], self.X[self.Y == 3,3], marker = 'o', color = 'g',label = 'virginica')

        axs[0, 0].set(ylabel = 'sepal width')
        axs[1, 0].set(xlabel = 'sepal length')
        axs[1, 0].set(ylabel = 'petal width')
        axs[1, 1].set(xlabel = 'petal length')
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
            params = { 'kernel': ['linear'], 'kernelparameter': [1], 'regularization': [None]}#np.logspace(-2,4, 200)*len(y_tr_r) }
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
            params = { 'kernel': ['gaussian'], 'kernelparameter': np.logspace(-2,5,100), 'regularization': [None] }
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
            params = { 'kernel': ['polynomial'], 'kernelparameter': np.logspace(-2,1,25), 'regularization': [None] }
            cvsvm = cv(self.X_tr, y_tr_r, svm_qp, params, loss_function=zero_one_loss, nfolds=5)
            print('Target class: {}'.format(target))
            print('Loss: {}'.format(cvsvm.cvloss))
            print('Parameters: {}'.format(cvsvm.params))
            accuracy.append(np.sum(np.sign(cvsvm.predict(self.X_te)) == y_te_r) / len(y_te_r))
        print('accuracy: {}'.format(accuracy))