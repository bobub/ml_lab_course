""" ps3_application.py

PUT YOUR NAME HERE:
<FIRST_NAME><LAST_NAME>


Write the functions
- roc_curve
- krr_app
(- roc_fun)

Write your code in the given functions stubs!


(c) Daniel Bartz, TU Berlin, 2013
"""
import numpy as np
import pylab as pl
import random
import matplotlib as plt
# from mpl_toolkits.mplot3d import Axes3D
# from matplotlib.lines import Line2D
from scipy.stats import norm
import os
import scipy.io as sio
import pickle


import ps3_implementation as imp
imp = reload(imp)
# ASSIGNMENT 3 CLASS 
# contains all the the excercises of assignment 3
class assignment3():
    """
    the object contains the tasks for assignment 3 of the ML Learning Course Excercise 3 
    """
    def __init__(self):
        """
        load data for assignment 3
        CAVEAT path to data might be different 
        """
        # load data 
        cwd = os.getcwd()
        file_name = 'qm7.mat'
        path_to_data = cwd + '/data/' + xtrain + '/data/'+file_name
        assert os.path.exists(path_to_data), "The path does not excist."
        data = sio.loadmat(path_to_data)
        data_X = data['X'] # of shape 7165x23x23 
        self.y = data['T'].T
        # transform each msteolecule into a vector xi of R23 by eigenvalues of the M of R 23x23
        self.X, eigenvec = np.linalg.eigh(data_X)

        # Visualize data
        #plt.scatter(data['X'], data['R'])
    def distance_plot(self):
        """ 
        Excercise 3a
        calculate pairwise distance of the dataset
        """
        # calculate pairwise distance
        D = np.linalg.norm(self.X[None,:]-self.X[:,None], axis = 2)
        # calculate absolute energy difference
        y_abs_diff = abs(self.y[None,:,:] - self.y[:,:,None])
        # plot
        plt.figure(figsize=(8, 8))
        plt.plot(D,y_abs_diff.reshape(len(self.y),len(self.y)),'bo');
        plt.xlabel("||$x_{i}$ - $x_{j}$||")
        plt.ylabel("|$y_{i}$ - $y_{j}$|")
        
    def split_data(self):
        """
        # Excercise 3b
        Split data randomly into training set of 5000 and test of 2165 samples
        seed = 2
        """
        # split data
        # Random Partitioning
        X_pos = np.linspace(0,len(self.X)-1, len(self.X))
        random.Random(2).shuffle(X_pos)
        self.Xtr = self.X[X_pos[:5000].astype('int')]
        self.Xte = self.X[X_pos[5000:].astype('int')]
        self.Ytr = self.y[X_pos[:5000].astype('int')]
        self.Yte = self.y[X_pos[5000:].astype('int')]

    def fold_cross_validation(self):
        """
        # Excercise 3c
        Perform a cross validation on a training set of 2500
        Perform split_data before
        """
        # Fivefold Cross validation
        train_samples = random.sample(range(0, 5000), 2500)
        Xtr2500 = self.Xtr[train_samples]
        Ytr2500 = self.Ytr[train_samples]
        D2500= np.linalg.norm(Xtr2500[None,:]-Xtr2500[:,None], axis = 2)
        quantiles = np.quantile(D2500,[0.1,0.5, 0.9])
        params = { 'kernel': ['gaussian'], 'kernelparameter': quantiles, 'regularization': np.logspace(-7,0, 10) }
        self.cvkrr = imp.cv(Xtr2500, Ytr2500, imp.krr, params, loss_function=mean_absolute_error, nfolds=5)
        y_pred2500 = self.cvkrr.predict(self.Xte)
        MAE = mean_absolute_error(self.Yte, y_pred2500)
        print("The mean absolute error is: {} ".format(round(MAE,2)))
        print("The best regularzation parameter C is: {}".format(self.cvkrr.regularization))
        print("The best kernelparameter sigma is: {}".format(self.cvkrr.kernelparameter))

    def plot_MAE_for_different_nsamples(self):
        """
        # Excercise 3d
        Plot MAE for different nsamples
        Perform fold_cross_validation before
        """
        MAE = []
        n_samples = [100,300,600,900,1200,1700,2000,2700,3000,3900,4200,4500,4700,4800,4900,4950,5000]
        for i in tqdm(n_samples):
            train_samples = random.sample(range(0, 5000), i)
            Xtr_nsample = self.Xtr[train_samples]
            Ytr_nsample = self.Ytr[train_samples]
            model = imp.krr([self.cvkrr.kernel][0], [self.cvkrr.kernelparameter][0], [self.cvkrr.regularization])
            model.fit(Xtr_nsample, Ytr_nsample)
            y_pred = model.predict(self.Xte)
            MAE.append(mean_absolute_error(self.Yte,y_pred))
        plt.figure(figsize =(8,6))
        plt.plot(n_samples, MAE, 'bo')
        plt.xlabel("n training samples")
        plt.ylabel("Mean Absolute Error [kcal/mol]")
        
    def plot_energies_for_1000(self):
        """ 
        Excercise 3e, perform under-, well- and overfit for 1000 training samples
        """
        # split data
        # Random Partitioning
        X_pos = np.linspace(0,len(self.X)-1, len(self.X))
        random.Random(4).shuffle(X_pos)
        Xtr1000 = self.X[X_pos[:1000].astype('int')]
        Xte1000 = self.X[X_pos[1000:].astype('int')]
        Ytr1000 = self.y[X_pos[:1000].astype('int')]
        Yte1000 = self.y[X_pos[1000:].astype('int')]
        
        # get parameter for good fit
        # Fivefold Cross validation

        D1000= np.linalg.norm(Xtr1000[None,:]-Xtr1000[:,None], axis = 2)
        quantiles = np.quantile(D1000,[0.1,0.5, 0.9])
        params = { 'kernel': ['gaussian'], 'kernelparameter': quantiles, 'regularization': np.logspace(-7,0, 10) }
        cvkrr = imp.cv(Xtr1000, Ytr1000, imp.krr, params, loss_function=mean_absolute_error, nfolds=5)
        y_pred1000 = cvkrr.predict(Xte1000)
        MAE = mean_absolute_error(Yte1000, y_pred1000)
        
        # result of CV
        print("The mean absolute error is: {} ".format(round(MAE,2)))
        print("The best regularzation parameter C is: {}".format(cvkrr.regularization))
        print("The best kernelparameter sigma is: {}".format(cvkrr.kernelparameter))
        print("The cvloss: {}".format(cvkrr.cvloss))
        
        # define parameters for training
        params = { 'kernel': ['linear','gaussian', 'gaussian'], 'kernelparameter': [False, cvkrr.kernelparameter, 1], 'regularization': [cvkrr.regularization, cvkrr.regularization, 0] }
        
        # plot
        plt.figure(figsize =(10,6))
        for i in [0,1,2]:
            model = imp.krr(params['kernel'][i], params['kernelparameter'][i], params['regularization'][i])
            model.fit(Xtr1000,Ytr1000)
            y_pred_train = model.predict(Xtr1000)
            y_pred = model.predict(self.Xte)
            plt.subplot(1,3,i+1)
            plt.plot(self.Yte,y_pred,'bo')
            plt.plot(Ytr1000,y_pred_train,'ro')
            plt.xlabel("y_true")
            plt.ylabel("y_pred")
            plt.legend(labels = ['test', 'train'])
        plt.tight_layout(pad=3.0)

assign = assignment3()
# Excecise 3a - takes long
assign.distance_plot()
# Excecise 3b
assign.split_data()
# Excecise 3c
assign.fold_cross_validation()
# Excecise 3d
assign.plot_MAE_for_different_nsamples()
# Excecise 3e
assign.plot_energies_for_1000()

# ASSIGNMENT 4 IMPLEMENTATION FUNCTIONS
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


# 4a
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

# This functions implements all of assignment 4. It is copied from various cells in a Jupyter Notebook.
# It is not designed for .py environment but serves as a record of methodology.
# If required, the original .jpynb can be requested.
def assignment_4():
    # 4b
    # load data
    import pandas as pd
    cwd = os.getcwd()

    xtrain_names = ['U04_banana-xtrain.dat', 'U04_diabetis-xtrain.dat', 'U04_flare-solar-xtrain.dat',
                    'U04_image-xtrain.dat', 'U04_ringnorm-xtrain.dat']
    ytrain_names = ['U04_banana-ytrain.dat', 'U04_diabetis-ytrain.dat', 'U04_flare-solar-ytrain.dat',
                    'U04_image-ytrain.dat', 'U04_ringnorm-ytrain.dat']
    xtest_names = ['U04_banana-xtest.dat', 'U04_diabetis-xtest.dat', 'U04_flare-solar-xtest.dat', 'U04_image-xtest.dat',
                   'U04_ringnorm-xtest.dat']
    ytest_names = ['U04_banana-ytest.dat', 'U04_diabetis-ytest.dat', 'U04_flare-solar-ytest.dat', 'U04_image-ytest.dat',
                   'U04_ringnorm-ytest.dat']

    xtrain_data = []
    ytrain_data = []
    xtest_data = []
    ytest_data = []

    all_datasets = ['banana', 'diabetis', 'flare-solar', 'image', 'ringnorm']

    folds = [10, 9, 9, 10, 10]

    for (xtrain, ytrain, xtest, ytest) in zip(xtrain_names, ytrain_names, xtest_names, ytest_names):
        path_to_data = cwd + '/data/' + xtrain
        assert os.path.exists(path_to_data), "The path does not exist."
        xtrain_data.append(np.loadtxt(path_to_data))

        path_to_data = cwd + '/data/' + ytrain
        assert os.path.exists(path_to_data), "The path does not exist."
        ytrain_data.append(np.loadtxt(path_to_data))

        path_to_data = cwd + '/data/' + xtest
        assert os.path.exists(path_to_data), "The path does not exist."
        xtest_data.append(np.loadtxt(path_to_data))

        path_to_data = cwd + '/data/' + ytest
        assert os.path.exists(path_to_data), "The path does not exist."
        ytest_data.append(np.loadtxt(path_to_data))

    # 4b - GENERATE DICTIONARY RESULTS FOR EACH DATASET

    params = {'kernel': ['linear', 'polynomial'], 'kernelparameter': [1, 2, 3], 'regularization': [0]}

    results = {'banana': {'cvloss': [0], 'kernel': [0], 'kernelparameter': [0], 'regularization': [0], 'y_pred': [0]},
               'diabetis': {'cvloss': [0], 'kernel': [0], 'kernelparameter': [0], 'regularization': [0], 'y_pred': [0]},
               'flare-solar': {'cvloss': [0], 'kernel': [0], 'kernelparameter': [0], 'regularization': [0],
                               'y_pred': [0]},
               'image': {'cvloss': [0], 'kernel': [0], 'kernelparameter': [0], 'regularization': [0], 'y_pred': [0]},
               'ringnorm': {'cvloss': [0], 'kernel': [0], 'kernelparameter': [0], 'regularization': [0], 'y_pred': [0]}}

    # bug description - "setting an array element with a sequence" if len(xtrain_data) is not equally divisible by nfolds
    # solving the bug is very difficult, because it would require converting the unequal sequences into numpy arrays
    # and filling the missing values. These values are then indexed on the training data, and will result in an error or a datapoint being used repeatedly
    # depending on how you choose to fill the values

    # so the obvious solution is to pick n_folds so that len(xtrain)%n_folds=0 ie n_folds is a multiple of xtrain
    for (xtrain, ytrain, xtest, ytest, dataset, fold) in zip(xtrain_data, ytrain_data, xtest_data, ytest_data,
                                                             all_datasets, folds):
        print('Xtrain\n', xtrain.shape)
        print('ytrain\n', ytrain.shape)
        print('Xtest\n', xtest.shape)
        print('ytest\n', ytest.shape)

        cvkrr = cv(xtrain.T, ytrain, krr, params, loss_function=zero_one_loss, nfolds=fold, nrepetitions=5)
        y_pred = cvkrr.predict(xtest.T)

        results[dataset]['y_pred'] = y_pred
        results[dataset]['kernel'] = cvkrr.kernel
        results[dataset]['kernelparameter'] = cvkrr.kernelparameter
        results[dataset]['regularization'] = cvkrr.regularization
        results[dataset]['cvloss'] = cvkrr.cvloss

    params = {'kernel': ['linear', 'gaussian'], 'kernelparameter': [0.1, 0.5, 0.9], 'regularization': [0]}

    for (xtrain, ytrain, xtest, ytest, dataset, fold) in zip(xtrain_data, ytrain_data, xtest_data, ytest_data,
                                                             all_datasets, folds):
        print('Xtrain\n', xtrain.shape)
        print('ytrain\n', ytrain.shape)
        print('Xtest\n', xtest.shape)
        print('ytest\n', ytest.shape)

        cvkrr = cv(xtrain.T, ytrain, krr, params, loss_function=zero_one_loss, nfolds=fold, nrepetitions=5)
        y_pred = cvkrr.predict(xtest.T)

        if results[dataset]['cvloss'] > cvkrr.cvloss:
            results[dataset]['y_pred'] = y_pred
            results[dataset]['kernel'] = cvkrr.kernel
            results[dataset]['kernelparameter'] = cvkrr.kernelparameter
            results[dataset]['regularization'] = cvkrr.regularization
            results[dataset]['cvloss'] = cvkrr.cvloss

    # manually remove kernelparameter from linear soln.
    results['flare-solar']['kernelparameter'] = None

    # open a file, where you want to store the data
    file = open('results.p', 'wb')

    # dump information to that file
    pickle.dump(results, file)

    # close the file
    file.close()

    #4C - PLOT ROC CURVES FOR VARYING BIASES

    for (xtrain, ytrain, dataset, fold) in zip(xtrain_data, ytrain_data, all_datasets, folds):
        print('Xtrain\n', xtrain.shape)
        print('ytrain\n', ytrain.shape)

        params = {'kernel': [str(results[dataset]['kernel'])],
                  'kernelparameter': [(results[dataset]['kernelparameter'])],
                  'regularization': [(results[dataset]['regularization'])]}
        # print(params['kernel'])
        cvkrr = cv(xtrain.T, ytrain, krr, params, loss_function=roc_fun, nfolds=fold, nrepetitions=4)

        loss = cvkrr.cvloss
        # print('fpr\n',loss[0])
        # print('tpr\n',loss[1])
        # print(loss)

        fpr = np.append(loss[0], 0)
        fpr = np.insert(fpr, 0, 1)

        tpr = np.append(loss[1], 0)
        tpr = np.insert(tpr, 0, 1)

        # plot ROC fun
        plt.figure(figsize=(4.5, 4.5))
        plt.plot(fpr, tpr, label='KRR algorithm')
        plt.plot(np.arange(0, 1.1, 0.1), np.arange(0, 1.1, 0.1), label='Random guess')
        plt.ylabel('True Positive Rate (TPR)')
        plt.xlabel('False Positive Rate (FPR)')
        plt.title('%s dataset\'s average ROC curve from a varying bias' % dataset)
        plt.legend()

    # 4.d - COMPARE LOOCV TO CV REGULARISATION
    cv_regularisation = []

    for (xtrain, ytrain, xtest, ytest, dataset, fold) in zip(xtrain_data, ytrain_data, xtest_data, ytest_data,
                                                             all_datasets, folds):
        print('Xtrain\n', xtrain.shape)
        print('ytrain\n', ytrain.shape)
        print('Xtest\n', xtest.shape)
        print('ytest\n', ytest.shape)

        params = {'kernel': [results[dataset]['kernel']], 'kernelparameter': [results[dataset]['kernelparameter']],
                  'regularization': np.logspace(-5, 5, 11)}

        cvkrr = cv(xtrain.T, ytrain, krr, params, loss_function=zero_one_loss, nfolds=fold, nrepetitions=5)
        y_pred = cvkrr.predict(xtest.T)

        cv_regularisation.append(cvkrr.cvloss)
