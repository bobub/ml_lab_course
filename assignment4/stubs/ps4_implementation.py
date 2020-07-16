""" ps4_implementation.py

PUT YOUR NAME HERE:
<FIRST_NAME><LAST_NAME>


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
import torch



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

        
        # calculate kernelmatrix
        #if self.kernel == 'linear':
        #    self.__linearKernel(X)
        #elif self.kernel == 'polynomial':
        #    self.__polynomialKernel(X)
        #elif self.kernel == 'gaussian':
        #    self.__gaussianKernel(X)
        #else:
        # print("""The following kernel {} is not known. Please use either 'linear' , 'polynomial' or 'gaussian'.""".format(kernel))

        self.kernelmatrix = buildKernel(self.X_sv.T, kernel = self.kernel, kernelparameter = self.kernelparameter)
        
        P = (self.Y_sv@self.Y_sv.T)*self.kernelmatrix
        q = -1*np.ones((self.__ydim,1)) #y_train has the same length as X_train
        A = self.Y_sv.T
        b = 0
        if self.C is None: 
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
            h = np.hstack((np.zeros(self.__ydim),self.C*np.ones(self.__ydim)))
        # this is already implemented so you don't have to
        # read throught the cvxopt manual
        alpha = np.array(qp(cvxmatrix(P, tc='d'),
                            cvxmatrix(q, tc='d'),
                            cvxmatrix(G, tc='d'),
                            cvxmatrix(h, tc='d'),
                            cvxmatrix(A, tc='d'),
                            cvxmatrix(b, tc='d'))['x']).flatten()
        # Support vectors have non zero lagrange multipliers and are smaller than C/m
        self.sv = np.logical_and(alpha > 1e-5, alpha < np.round(self.C,5)) #treshold 1e-5
        self.alpha_sv = alpha[self.sv]
        
        self.X_sv = self.X_sv[self.sv]
        self.Y_sv = self.Y_sv[self.sv]
        # calculation of bias b
        
        # calculate kernelmatrix for X_sv, makes it independet of the choosen kernel
        
        #if self.kernel == 'linear':
        #    self.__linearKernel(self.X_sv)
        #elif self.kernel == 'polynomial':
        #    self.__polynomialKernel(self.X_sv)
        #elif self.kernel == 'gaussian':
        #    self.__gaussianKernel(self.X_sv)
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
        
        # calculate kernelmatrix
        #if self.kernel == 'linear':
        #    self.__linearKernel(X)
        #elif self.kernel == 'polynomial':
        #    self.__polynomialKernel(X)
        #elif self.kernel == 'gaussian':
        #    self.__gaussianKernel(X)

        self.kernelmatrix = buildKernel(self.X_sv.T,X.T, kernel = self.kernel, kernelparameter = self.kernelparameter)
        self.yhat = np.sign((self.alpha_sv.reshape(-1,1)*self.Y_sv).T @ self.kernelmatrix + self.b)

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