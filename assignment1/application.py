""" sheet1_implementation.py

PUT YOUR NAME HERE:
Boris Bubla
Leonard Paeleke


NOTE:
The application assignments were originally solved using jupyter notebook.
Those solutions were encapsulated in the functions presented here.
Thus, many values are hardcoded and the functions are not particularly reusable.
They have been submitted as evidence of methodology and approach, should the results in the report.pdf not be correct.

(c) Daniel Bartz, TU Berlin, 2013
"""
import numpy as np
import pylab as pl
import random
import implementation.py as imp
from mpl_toolkits.mplot3d import Axes3D
import matplotlib
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec
import scipy.io as sio
import os
import scipy.linalg as la
import scipy.spatial as sp
from scipy.linalg import expm

imp = reload(imp)

"""
ASSIGNMENT 5 - USPS DATASET
"""
# Plot function for 2 b
def plot_fun(pca):
    """
    Plot function for Assignment 5
    Arranges the prinicple components plots
    """
    # intialize figure environment
    fig = plt.figure(figsize=(14, 8))
    gs = fig.add_gridspec(5, 3)

    # first plot: all principle components
    f_ax1 = fig.add_subplot(gs[:, 0])
    plt.bar(range(len(pca.D)), pca.D)
    plt.xticks(range(0, len(pca.D), 50), np.append(1, np.arange(50, len(pca.D), 50)))
    plt.title('all principle components')

    # second plot: first 25 principle components
    fig.add_subplot(gs[:, 1])
    plt.bar(range(25), pca.D[:25])
    plt.title('first 25 principle components')
    plt.xticks(range(0, 25, 6), range(1, 26, 6))

    # third plot: visualize first 5 principle components
    for i in range(5):
        f_ax3 = fig.add_subplot(gs[i, 2])
        if i == 0:
            f_ax3.set_title('first 5 components')
        plt.imshow(pca.U[i].reshape(16, 16), cmap='gray')

    plt.show()
# assignment 5 analysis
def assignment_5():

    # Load data
    cwd = os.getcwd()
    file_name = 'usps.mat'
    path_to_data = cwd + '/data/'+file_name
    assert os.path.exists(path_to_data), "The path to the data does not exist."

    data = sio.loadmat(path_to_data)

    data_labels = data['data_labels']
    data_patterns = data['data_patterns']



    # perform pca
    pca = imp.PCA(data_patterns.T)
    plot_fun(pca)

    # Plot 3a).
    # generate noisy data
    sigmas = [0.05, 0.3, 1]
    length, width = data_patterns.T.shape
    counter = 1
    # sigmas = [1]
    for sigma in sigmas:

        data_patterns_noisy = np.copy(data_patterns.T)  # prevent same disk space of variables
        # principle component evaluation
        if counter != 3:
            # noisy data
            length, width = data_patterns.T.shape
            data_patterns_noisy = sigma * np.random.randn(length, width) + data_patterns_noisy
        else:
            # outlier data
            pics = [0, 1, 2, 13, 34]
            length, width = data_patterns_noisy[[0, 1, 2, 13, 34]].shape
            # add gaussian noise to only 5 data points
            data_patterns_noisy[[0, 1, 2, 13, 34]] = sigma * np.random.randn(length, width) + data_patterns_noisy[
                [0, 1, 2, 13, 34]]

        pca_noise = imp.PCA(data_patterns_noisy)
        plot_fun(pca_noise)

        # reconstruction by m principle components
        data_patterns_pjt = pca_noise.denoise(data_patterns_noisy, 13)

    # 3b/c).
    # generate noisy data
    sigmas = [0, 0.05, 0.05, 0.05, 0.3, 0.3, 0.3, 1, 1, 1]
    img_num = [5, 69, 68, 2, 13, 121, 1, 34, 70, 0]  # 0 - 9
    m = [7, 7, 25, 50, 7, 25, 50, 7, 25, 50]
    length, width = data_patterns.T.shape

    fig = plt.figure(figsize=(8, 9))
    gs = fig.add_gridspec(10, 3)
    for idx, sigma in enumerate(sigmas):

        data_patterns_noisy = np.copy(data_patterns.T)  # prevent same disk space of variables
        # principle component evaluation
        if sigma != 1:
            # noisy data
            length, width = data_patterns.T.shape
            data_patterns_noisy = sigma * np.random.randn(length, width) + data_patterns_noisy
        else:
            # outlier data
            pics = [0, 1, 2, 13, 34]
            length, width = data_patterns_noisy[[0, 1, 2, 13, 34]].shape
            # add gaussian noise to only 5 data points
            data_patterns_noisy[[0, 1, 2, 13, 34]] = sigma * np.random.randn(length, width) + data_patterns_noisy[
                [0, 1, 2, 13, 34]]

        pca_noise = imp.PCA(data_patterns_noisy)
        # plot_fun(pca_noise)

        # reconstruction by m principle components
        data_patterns_pjt = pca_noise.denoise(data_patterns_noisy, m[idx])
        # img_num = 0
        f_ax1 = fig.add_subplot(gs[idx, 0])
        plt.imshow(data_patterns.T[img_num[idx]].reshape(16, 16), cmap='gray')
        f_ax1 = fig.add_subplot(gs[idx, 1])
        plt.imshow(data_patterns_noisy[img_num[idx]].reshape(16, 16), cmap='gray')
        f_ax1 = fig.add_subplot(gs[idx, 2])
        plt.imshow(data_patterns_pjt[img_num[idx]].reshape(16, 16), cmap='gray')

"""
ASSIGNMENT 6 - BANANA DATASET / OUTLIER DETECTION ALGORITHMS
"""

# performs outlier detection performance evaluation for loops number of times.
# loops currently hardcoded to 1 --- this has all been adapted from a jupyter notebook.
def outliers_calc():
    ''' outlier analysis for assignment 6 and displays the results'''


    # load and unpack data
    cwd = os.getcwd()
    file_name = 'banana.npz'
    path_to_data = cwd + '/data/'+file_name
    assert os.path.exists(path_to_data), "The path do the data does not exist."

    data = np.load(path_to_data)

    # data manipulation and preprocessing
    outlier_rate=np.array([0.01,0.1,0.5,1])
    pos_idx=np.argwhere(data['label']==1).T[1]# idx of pos class
    n_outliers=np.round(outlier_rate*len(pos_idx))
    pos_data=data['data'].T[pos_idx]
    n_outliers=n_outliers.astype(int)

    # This takes about 30min to run for 100 loops --- inefficient but gets the job done.
    # Loops is now set to 1 for convenience
    loops = 1

    results_3 = np.empty((loops, len(n_outliers)))
    results_10 = np.empty((loops, len(n_outliers)))
    results_mean = np.empty((loops, len(n_outliers)))

    for i in range(loops):

        for j in n_outliers:
            # draw outliers from uniform box
            outliers = np.random.uniform(low=-4, high=4, size=(j, 2))
            #print(outliers[0][0])

            # reset data
            data_ = data['data'].T
            labels_ = data['label'].T

            # add outliers to positive class
            data_ = np.append(data_, outliers, axis=0)
            labels_ = np.append(labels_, np.ones((j, 1)), axis=0)

            # compute gamma 3,10,mean
            gamma_3 = imp.gammaidx(data_, 3)
            gamma_10 = imp.gammaidx(data_, 10)
            # compute dist_to_mean
            diff = np.subtract(data_, np.mean(data_, axis=0))
            dist_to_mean = np.linalg.norm(diff, axis=1)

            # compute auc
            auc_3 = imp.auc(labels_, gamma_3)
            auc_10 = imp.auc(labels_, gamma_10)
            auc_mean = imp.auc(labels_, dist_to_mean)

            # store results
            results_3[i][np.argwhere(n_outliers == j)] = auc_3[0]
            results_10[i][np.argwhere(n_outliers == j)] = auc_10[0]
            results_mean[i][np.argwhere(n_outliers == j)] = auc_mean[0]

    # plot the results
    fig, ax = plt.subplots(figsize=(12,12))
    ax.set_xlabel('Percentage of outliers added relative to positive class (%)')
    ax.set_ylabel('AUC Score')
    bp1=ax.boxplot(results_10,patch_artist=True,boxprops=dict(facecolor='green', color='green'))
    bp2=ax.boxplot(results_3,patch_artist=True,boxprops=dict(facecolor='red', color='red'))
    bp3=ax.boxplot(results_mean,patch_artist=True,boxprops=dict(facecolor='blue', color='blue'))
    ax.set_xticks([1, 2, 3, 4])
    ax.set_xticklabels(['1', '10', '50', '100'], fontsize=12)
    ax.legend([bp1["boxes"][0], bp2["boxes"][0], bp3["boxes"][0]], ['$\gamma$-index with k=10', '$\gamma$-index with k=3','Distance to mean'], loc='upper left')

# performs exemplary plot at 50% contamination rate
def outliers_exemplary():
    ''' performs an exemplary run and plots the results'''

    # exemplary run
    n_outliers = np.array([1188])  # 50% contamination

    # draw outliers from uniform box
    outliers = np.random.uniform(low=-4, high=4, size=(n_outliers[0], 2))

    # pos data
    pos = data['data'].T

    # add outliers to positive class
    data_ = np.append(pos, outliers, axis=0)

    # compute gamma 3,10,mean
    gamma_3 = imp.gammaidx(data_, 3)
    gamma_10 = imp.gammaidx(data_, 10)
    # compute dist_to_mean
    diff = np.subtract(data_, np.mean(data_, axis=0))
    dist_to_mean = np.linalg.norm(diff, axis=1)

    # exemplary plots
    fig, axes = plt.subplots(nrows=1, ncols=3,figsize=(16,6))
    fig.suptitle('Marker sizes have been scaled by outlier scores', fontsize=12)

    axes[0].scatter(x=pos.T[0],y=pos.T[1],s=gamma_3[:5300]*100,c='r',label='data',alpha=0.3,edgecolors=None,)
    axes[0].scatter(x=outliers.T[0],y=outliers.T[1],s=gamma_3[5300:]*100,c='b',label='outliers',alpha=0.3,edgecolors=None)
    axes[0].set_xlabel('x')
    axes[0].set_ylabel('y')
    axes[0].set_title('gamma-index with k=3')
    axes[0].legend()

    axes[1].scatter(x=pos.T[0],y=pos.T[1],s=gamma_10[:5300]*100,c='r',label='data',alpha=0.3,edgecolors=None,)
    axes[1].scatter(x=outliers.T[0],y=outliers.T[1],s=gamma_10[5300:]*100,c='b',label='outliers',alpha=0.3,edgecolors=None)
    axes[1].set_xlabel('x')
    axes[1].set_ylabel('y')
    axes[1].set_title('gamma-index with k=10')
    axes[1].legend()

    axes[2].scatter(x=pos.T[0],y=pos.T[1],s=dist_to_mean[:5300]*10,c='r',label='data',alpha=0.3,edgecolors=None,)
    axes[2].scatter(x=outliers.T[0],y=outliers.T[1],s=dist_to_mean[5300:]*10,c='b',label='outliers',alpha=0.3,edgecolors=None)
    axes[2].set_xlabel('x')
    axes[2].set_ylabel('y')
    axes[2].set_title('distance to mean')
    axes[2].legend()


"""
ASSIGNMENT 7 - LLE APPLICATION 
"""
def lle_visualize():
    # load data
    cwd = os.getcwd()
    file_name = 'fishbowl_dense.npz'
    path_to_data = cwd + '/data/' + file_name
    assert os.path.exists(path_to_data), "The path does not excist."
    fishbowl = np.load(path_to_data)

    cwd = os.getcwd()
    file_name = 'swissroll_data.npz'
    path_to_data = cwd + '/data/' + file_name
    assert os.path.exists(path_to_data), "The path does not excist."
    swissroll = np.load(path_to_data)

    cwd = os.getcwd()
    file_name = 'flatroll_data.npz'
    path_to_data = cwd + '/data/' + file_name
    assert os.path.exists(path_to_data), "The path does not excist."
    flatroll = np.load(path_to_data)

    # format data and references
    fishbowl_data = fishbowl['X'].T
    fishbowl_ref = fishbowl['X'].T[:, 2]
    swissroll_data = swissroll['x_noisefree'].T
    swissroll_ref = swissroll['z'].T[:, 0]
    flatroll_data = flatroll['Xflat'].T
    flatroll_ref = flatroll['true_embedding'].T

    # FISHBOWL
    # apply lle
    fishbowl_lle = imp.lle(fishbowl_data, 2, 0.00001, 'eps-ball',
                       epsilon=0.29)  # k=50-80 is good, epsilon=0.27 is best so far

    # plot

    fig = plt.figure(figsize=(14, 6))

    # normalize flatroll_ref -> values from 0 to 1
    color_code = ((fishbowl_ref - min(fishbowl_ref)) / (max(fishbowl_ref) - min(fishbowl_ref))).reshape(
        len(fishbowl_ref), )
    ax = fig.add_subplot(1, 2, 1, projection='3d')
    ax.scatter(xs=fishbowl_data.T[0], ys=fishbowl_data.T[1], zs=fishbowl_data.T[2], s=1, c=color_code, cmap='hsv')
    ax.set_xlabel('X1')
    ax.set_ylabel('X2')
    ax.set_zlabel('X3')
    ax.set_xticks([-1, -0.5, 0, 0.5, 1])
    ax.set_yticks([-1, -0.5, 0, 0.5, 1])
    ax.set_zticks([-1, -0.5, 0, 0.5, 1])
    ax.set_title('Fishbowl in 3D')
    ax = fig.add_subplot(1, 2, 2)
    ax.scatter(x=fishbowl_lle.T[0], y=fishbowl_lle.T[1], s=1, c=color_code, cmap='hsv')
    ax.set_xlabel('X1')
    ax.set_ylabel('X2')
    ax.set_title('Fishbowl in 2D')

    # SWISSROLL
    # apply lle
    swissroll_lle = imp.lle(swissroll_data, 2, 0.000001, 'eps-ball', epsilon=5)  # k=60 good, epsilon=7 good

    # plot
    fig = plt.figure(figsize=(14, 6))

    # normalize flatroll_ref -> values from 0 to 1
    color_code = ((swissroll_ref - min(swissroll_ref)) / (max(swissroll_ref) - min(swissroll_ref))).reshape(
        len(swissroll_ref), )
    ax = fig.add_subplot(1, 2, 1, projection='3d')
    ax.scatter(xs=swissroll_data.T[0], ys=swissroll_data.T[1], zs=swissroll_data.T[2], s=4, c=color_code, cmap='hsv')
    ax.set_xlabel('X1')
    ax.set_ylabel('X2')
    ax.set_zlabel('X3')
    ax.set_title('Swissroll in 3D')
    ax = fig.add_subplot(1, 2, 2)
    ax.scatter(x=swissroll_lle.T[0], y=swissroll_lle.T[1], s=4, c=color_code, cmap='hsv')
    ax.set_xlabel('X1')
    ax.set_ylabel('X2')
    ax.set_title('Swissroll in 2D')

    # FLATROLL
    # apply lle
    flatroll_lle = imp.lle(flatroll_data, 1, 0.00001, 'knn', k=9)

    # plot
    fig = plt.figure(figsize=(18, 8))
    # normalize flatroll_ref -> values from 0 to 1
    color_code = ((flatroll_ref - min(flatroll_ref)) / (max(flatroll_ref) - min(flatroll_ref))).reshape(
        len(flatroll_ref), )
    ax = fig.add_subplot(1, 2, 1)
    ax.scatter(x=flatroll_data.T[0], y=flatroll_data.T[1], s=3, c=color_code, cmap='hsv')
    ax.set_xlabel('X1')
    ax.set_ylabel('X2')
    ax.set_title('Flatroll in 2D')
    ax = fig.add_subplot(1, 2, 2)
    ax.scatter(x=flatroll_ref.reshape(len(flatroll_ref)), y=flatroll_lle.T, s=3, c=color_code, cmap='hsv')
    ax.set_xlabel('flatroll_ref')
    ax.set_ylabel('flatroll_lle')
    ax.set_title('Flatroll in 1D')




"""
ASSIGNMENT 8 - FLATROLL LLE
"""
#knn plot function used later on
def plot_knn(X, k):
    """
    Generates a neighborhood graph from a 2D data set
    """
    D = np.sqrt(np.sum((X[None, :] - X[:, None])**2, -1))
    kn = np.argsort(D,kind='mergesort')
    # identify k-nearest neighbors
    kn = kn[:,:k+1]
    length, width = kn.shape
    for i in range(length):
        for j in range(width-1):
            plt.plot(flatroll_data[kn[i][[0,j]]][:,0],flatroll_data[kn[i][[0,j]]][:,1], 'k-o')

# plot function for assignment 8
def plot_8(flatroll_data, flatroll_ref, flatroll_lle, k):
    # normalize flatroll_ref -> values from 0 to 1
    color_code = ((flatroll_ref - min(flatroll_ref)) / (max(flatroll_ref) - min(flatroll_ref))).reshape(
        len(flatroll_ref), )

    fig = plt.figure(figsize=(18, 8))

    ax = fig.add_subplot(1, 2, 1)
    plot_knn(flatroll_data, k)
    ax.set_xlabel('X1')
    ax.set_ylabel('X2')
    ax.set_title('Flatroll in 2D')

    ax = fig.add_subplot(1, 2, 2)
    y = np.zeros((len(flatroll_data),))
    ax.scatter(x=flatroll_ref.reshape(len(flatroll_ref), ), y=flatroll_lle.T, s=3, c=color_code, cmap='hsv')

    ax.set_xlabel('flattroll_ref')
    ax.set_ylabel('X2')

    ax.set_title('Flatroll in 1D')

#assignment 8
def flatroll_lle():
    # FLATROLL
    # load data
    cwd = os.getcwd()
    file_name = 'flatroll_data.npz'
    path_to_data = cwd + '/data/' + file_name
    assert os.path.exists(path_to_data), "The path does not excist."
    flatroll = np.load(path_to_data)
    flatroll_data = flatroll['Xflat'].T
    flatroll_ref = flatroll['true_embedding'].T

    #add noise
    sigmas = [0.2, 0.2, 1.8, 1.8]
    k = [9, 50, 9, 50]
    length, width = flatroll_data.shape

    for idx, sigma in enumerate(sigmas):
        flatroll_data_noise = sigma * np.random.randn(length, width) + flatroll_data
        flatroll_lle = imp.lle(flatroll_data_noise, 1, 1e-6, 'knn', k=k[idx])
        plot_8(flatroll_data_noise, flatroll_ref, flatroll_lle, k[idx])

