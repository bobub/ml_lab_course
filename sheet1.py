""" sheet1_implementation.py

PUT YOUR NAME HERE:
Boris Bubla


Write the functions
- pca
- gammaidx
- lle
Write your implementations in the given functions stubs!


(c) Daniel Bartz, TU Berlin, 2013
"""
import numpy as np
import scipy.linalg as la
import scipy.spatial as sp

class PCA():  # Algorithms in guide.pdf used, pg 15-17
    def __init__(self, Xtrain):
        # Alg 1: Compute principal components
        self.C = np.cov(Xtrain)
        self.D, self.U = la.eigh(self.C)  # eigenvalues,eigenvectors

        # arrange in descending order
        descending = np.flip(np.argsort(self.D))
        self.D = self.D[descending]
        self.U = self.U[:, descending]

        self.mean_ = np.mean(Xtrain, axis=0)  # mean

    def project(self, Xtest, m):
        # Alg 2: Project to low dim space
        Z = np.empty((Xtest.shape[0], m))  # create output array
        X_center = np.subtract(Xtest, self.mean_)  # center data
        for i in range(Xtest.shape[0]):  # loop through samples
            z = np.multiply(self.U[i][:m].T, X_center[i][:m]).T  # compute
            Z = np.append(Z, [z], axis=0)
        return Z

    def denoise(self, Xtest, m):
        Z = self.project(Xtest, m)
        # Alg 3: Reconstruct in original space after project to low space
        Y = np.empty(Xtest.shape)  # create output array
        for i in range(Xtest.shape[0]):  # loop through samples
            y = self.mean_ + Z[i][:m].dot(self.U[i][:m])  # compute
            Y = np.append(Y, [y], axis=0)
        return Y


def gammaidx(X, k):
    n, d = X.shape
    # Calculate distance matrix
    D = sp.distance_matrix(X, X)
    # y=np.empty((n,1))#create output array
    y = []
    # Find all k nearest neigbours
    for i in range(n):
        D_0 = np.asarray(D[i][:])  # array of distances relative to i'th data point
        point = D_0[i]  # value at i'th data point
        gamma = 0
        for j in range(k):  # repeat for k nearest neigbours
            idx = (np.abs(D_0 - point)).argmin()  # find nearest neighbour
            gamma += np.abs(D_0[idx] - point)  # add distance/k to gamma for that point
            D_0 = np.delete(D_0, idx)  # remove nearest neighbour from list
        gamma = gamma/5
        y = np.append(y, [gamma], axis=0)

    return y


def lle(X, m, n_rule, param, tol=1e-2):
    ''' your header here!
    '''
    'Step 1: Finding the nearest neighbours by rule '
    
    'Step 2: local reconstruction weights'
    
    'Step 3: compute embedding'
