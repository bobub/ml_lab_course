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
import pickle


import ps3_implementation as imp
imp = reload(imp)

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
