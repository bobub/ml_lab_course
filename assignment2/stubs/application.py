"""
AUTHORS:
Boris Bubla
Leonard Paeleke

The application assignments were originally done in a Jupyter Notebook.
They have been converted to a .py file just for submission and are not intended designed for reusability.

It largely serves as evidence of methodology and approach when performing the application assignments.
"""

#imports
import os
import scipy.io as sio
import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
import implementation

# Assignment 7
"""
ASSIGNMENT 7: a few functions were edited for the purposes of this assignment. 
"""
def plot_gmm_solution_7(X,mu,sigma):
    """
    This function plots the different gaussians found by the EM algorithm as ellipses centred around the distributions' means.
    
    Input:
    X=data (nxd)
    mu=distribution centres (kxd)
    sigma=list of k dxd covariance matrices
    
    """
    #plot data points and setup plot parameters
    #plt.figure(figsize=(10,10))
    plt.scatter(X.T[0],X.T[1],s=20)
    plt.title('GMM solution found by EM algorithm with k = {}'.format(len(mu)))
    plt.ylabel('X2')
    plt.xlabel('X1')
    plt.grid(True)


    #draw ellipse
    for i,sig in enumerate(sigma):
        tline = np.linspace(0, 2 * np.pi, 100)
        sphere = np.vstack((np.sin([tline]), np.cos([tline])))
        ellipse = sqrtm(sig).dot(sphere)
        plt.plot(mu[i][0] + ellipse[0, :], mu[i][1] + ellipse[1, :],linewidth=4, color = 'k')
        #plot centre points
        plt.scatter(mu[i][0],mu[i][1],c='r',marker='x')
def em_gmm_7(X, k, max_iter=100, init_kmeans=False, tol=0.00001, converge_tol=0.0001):
    """
    This function applies the EM algorithm for Gaussian Mixture Models.
    It's adapted to investigate the quality as it has an extra output - iteration

    Inputs:
    X = data (nxd)
    k = number of gaussian components
    max_iter = the maximum amount of iterations attempted to find convergence
    init_kmeans = Initialises the EM algorithm using kmeans function, if True. Default is False.
    tol = The tolerance set for the convergence condition
    converge_tol = Tolerance for the convergence condition (optional)

    Outputs:
    pi = probability that a datapoint belongs to a cluster (1xk)
    mu = center points of clusters (kxd)
    sigma = list of k dxd covariance matrices
    loglik = the loglikehlihood at each iteration
    iteration = number of iterations until convergence
    """

    if init_kmeans == True:
        # 1.a INIT_KMEANS
        mu, r, _ = kmeans(X, k)
        unique, counts = np.unique(r, return_counts=True)
        pi = counts / np.sum(counts)

    else:
        # 1.b RANDOM INITIALISATIONS
        pi = np.full(shape=(k, 1), fill_value=1 / k)  # kx1
        rand_samples = np.random.choice(X.shape[0], size=(k,), replace=False)  # choose k random data points
        mu = X[rand_samples]  # centroid initialisation as random points, kxd

    # setup storage and loop
    sigma = [np.eye(X.shape[1]) for i in range(k)]  # dxd
    likelihoods = np.zeros(shape=(X.shape[0], k), dtype=float)  # nxk
    converged = False
    iteration = 1
    while (not converged) & (iteration <= max_iter):

        print('Iteration Number:\n', iteration)

        # 2. E-STEP - compute new likelihoods and responsibilities
        old_likelihoods = copy.deepcopy(likelihoods)
        # print('Old likelihoods\n', old_likelihoods)

        # 2.1 first find all k likelihoods
        for i in range(k):
            # nx1                             1x1 X nx1  = nx1
            likelihood = (pi[i] * norm_pdf(X, mu[i], sigma[i]))  # norm_pdf written to handle mu=(1xd) only
            likelihoods.T[i] = likelihood

        # CALC LOGLIK
        loglik = np.log(np.sum(likelihoods, axis=1)).sum()
        print('Loglikelihood\n', loglik)

        # 2.2 use likelihoods to calculate individual k responsibilities
        # nxk            nxk              nx1
        responsibilities = likelihoods / np.sum(likelihoods, axis=1).reshape(likelihoods.shape[0], 1)

        # 3. M-STEP - compute new n,pi,mu,sigma
        # 1xk
        n = np.sum(responsibilities, axis=0)
        # 1xk
        pi = n / np.sum(n, axis=0)
        # kxd                    (nxkx0)x(nx0xd)=nxkxd --> kxd / kx1
        mu = np.sum(responsibilities[:, :, None] * X[:, None, :], axis=0) / n.reshape(n.shape[0], 1)
        # kxdxd         =  sum ((nxkx0x0)     x    (nxkxdx0)x(nxkx0xd)) = nxkxdxd-->kxdxd/kx0x0
        sigma = np.sum(responsibilities[:, :, None, None] * (X[:, None, :, None] - mu[None, :, :, None]) * (
                    X[:, None, None, :] - mu[None, :, None, :]), axis=0) / n[:, None, None]
        #   (nx0xdx0-nxkx0x0)-->(nxkxdx0)
        # add regularisation term, tol
        sigma = sigma + tol * np.eye(X.shape[1])

        # break condition - only runs from second iteration to prevent log of old_likelihoods, which is 0 in iteration 1
        if iteration > 1:
            if (np.log(np.sum(old_likelihoods, axis=1)).sum() - loglik).all() < converge_tol:
                converged = True

        iteration = iteration + 1

    # return as a list of covariances
    list_sigma = [sigma[i, :, :] for i in range(k)]
    return pi, mu, list_sigma, loglik, iteration
def CCR(true_mean, data, function = 'k-means',iteration = 100):
    tp = 0
    fp = 0
    if np.isin(function, ['GMM','GMM_kmean_init']):
        iter_ = np.zeros(iteration)
    for i in range(iteration):
        if function == 'k-means':
            mu, r, loss = kmeans(X = data, k = 5)
        elif function == 'GMM':
            pi, mu, list_sigma, likelihoods, iter_[i] = em_gmm_7(X = data, k = 5)
        elif function == 'GMM_kmean_init':
            pi, mu, list_sigma, likelihoods, iter_[i] = em_gmm_7(X = data, k = 5, init_kmeans = True)
  
        for i in range(5):
            if (np.linalg.norm(true_mean[i]-mu, axis = 1)<0.1).any():
                tp+= 1
            else:
                fp+= 1
    try:
        plt.figure()
        plt.bar(range(iteration),iter_)
        plt.ylim(0,100)
    except:
        print('k-means is used')
    CCR = tp / (tp+fp) # correct precision rate
    print('CCR is: {}'.format(CCR))
    return CCR
    
def assignment_7():
    
    # load data 
    cwd = os.getcwd()
    file_name = '5_gaussians.npy'
    path_to_data = cwd + '/data/'+file_name
    assert os.path.exists(path_to_data), "The path does not excist."
    gaussians = np.load(path_to_data).T
    
    # Visualize data
    plt.scatter(gaussians.T[0], gaussians.T[1])
    
    # k-means on 5 gaussian data sets
    fig = plt.figure(figsize=(16, 9))
    gs = fig.add_gridspec(2, 3)
    for k in range(2,8):
        mu, r, loss = kmeans(X = gaussians, k = k)
        f_ax1 = fig.add_subplot(gs[int(k>=5), (k-2)%3])
        plt.title('k-means solution with k = {}'.format(len(mu)))
        plt.ylabel('X2')
        plt.xlabel('X1')
        plt.scatter(gaussians.T[0], gaussians.T[1])
        plt.scatter (mu.T[0], mu.T[1])
    
    # GMM on 5 gaussian data sets - without k-means initialization
    fig = plt.figure(figsize=(16, 9))
    gs = fig.add_gridspec(2, 3)
    for k in range(2,8):
        pi, mu, list_sigma, likelihoods = em_gmm(X = gaussians, k = k)
        f_ax1 = fig.add_subplot(gs[int(k>=5), (k-2)%3])
        plot_gmm_solution_7(X = gaussians ,mu = mu, sigma = list_sigma)
        
    # GMM on 5 gaussian data sets - with k-means initialization
    fig = plt.figure(figsize=(16, 9))
    gs = fig.add_gridspec(2, 3)
    for k in range(2,8):
        pi, mu, list_sigma, likelihoods = em_gmm(X = gaussians, k = k, init_kmeans = True)
        fig.add_subplot(2, 3, k-1) 
        plot_gmm_solution_7(X = gaussians ,mu = mu, sigma = list_sigma)
        
    # dendogram
    mu, r, loss =  kmeans(gaussians,15)
    R, kmloss, mergeidx = kmeans_agglo(gaussians,r)
    agglo_dendro(kmloss, mergeidx)
    # visualization of k=5 centroids found by hierachical clustering
    fig = plt.figure(figsize=(16, 9))
    for i in [2,3,4]:
        r = R[-i]
        mu = np.array([np.sum(gaussians[r == j], axis = 0) / (np.sum(r == j)) for j in np.unique(r)])
        plt.subplot(1,3,i-1)
        plt.title('k = {}'.format(len(mu)))
        plt.ylabel('X2')
        plt.xlabel('X1')
        plt.scatter(gaussians.T[0], gaussians.T[1])
        plt.scatter (mu.T[0], mu.T[1])
    
    # interpret quality of k-means, GMM, GMM_km
    true_mean = np.vstack((np.mean(gaussians[:100], axis = 0),np.mean(gaussians[100:200], axis = 0),np.mean(gaussians[200:300], axis = 0), np.mean(gaussians[300:400], axis = 0),np.mean(gaussians[400:], axis = 0)))
    # CCR 
    # k-means
    CCR_km = CCR(true_mean, gaussians, function = 'k-means')
    # GMM
    CCR_GMM = CCR(true_mean, gaussians, function = 'GMM')
    # GMM_km
    CCR_GMM_km = CCR(true_mean, gaussians, function = 'GMM_kmean_init')


# Assignment 8
def assignment_8():
    # load data
    cwd = os.getcwd()
    file_name = '2_gaussians.npy'
    path_to_data = cwd + '/data/' + file_name
    assert os.path.exists(path_to_data), "The path does not exist."
    gaussians = np.load(path_to_data).T

    # Visualize data
    plt.plot(gaussians[:, 0], gaussians[:, 1], 'o')
    plt.title('Plot of 2gaussians dataset')
    plt.ylabel('X1')
    plt.xlabel('X0')
    mu, r, loss = kmeans(X=gaussians, k=2)

    # apply kmeans 30 times to 2gaussians
    fig = plt.figure(figsize=(14, 85))
    gs = fig.add_gridspec(15, 2)
    for i in range(30):
        mu, r, loss = kmeans(X=gaussians, k=2)
        f_ax1 = fig.add_subplot(gs[i % 15, int(i >= 15)])
        plt.plot(gaussians[:, 0], gaussians[:, 1], 'o', ms=3, label='data')
        plt.plot(mu.T[0], mu.T[1], 'x', label='centroids')
        plt.ylabel('X1')
        plt.xlabel('X0')
        plt.title('2gaussians dataset with kmeans cluster centers')
        print(mu)

    # apply em_gmm 30 times to 2gaussians
    fig = plt.figure(figsize=(10, 80))
    gs = fig.add_gridspec(15, 2)
    for i in range(30):
        pi, mu, sigma, loglik = em_gmm(X=gaussians, k=2, tol=0.000001)
        print('mu', i, '\n', mu)
        f_ax1 = fig.add_subplot(gs[i % 15, int(i >= 15)])
        plt.plot(gaussians[:, 0], gaussians[:, 1], 'o', ms=3)
        plt.plot(mu.T[0], mu.T[1], 'x')
        plt.ylabel('X1')
        plt.xlabel('X0')
        plt.title('2gaussians dataset with gmm cluster centers')

    # apply em_gmm with kmeans initialisation 10 times to 2gaussians
    fig = plt.figure(figsize=(12, 85))
    gs = fig.add_gridspec(15, 2)
    for i in range(30):
        pi, mu, sigma, loglik = em_gmm(X=gaussians, k=2, init_kmeans=True, tol=0.0001)
        f_ax1 = fig.add_subplot(gs[i % 15, int(i >= 15)])
        plt.plot(gaussians[:, 0], gaussians[:, 1], 'o', ms=3)
        plt.plot(mu.T[0], mu.T[1], 'x')
        plt.ylabel('X1')
        plt.xlabel('X0')
        plt.title('Gmm cluster centers with init_kmeans')


#Assignment 9
def centroid_visualization(mu, title = 'plot'):
    """
    This function visualizes the centorids found in the usps data set, which contains images of size 16x16.
    The maximum number of centroids that can be visualized is 10.

    Inputs:
    mu = centroids of size 256x1
    title = title of the image
    
    Outputs:
    A figure containing 10 digits.
    """
    # results / Centroids Visualization of k-means 
    fig = plt.figure(figsize=(4, 9))
    plt.title(str(title))
    gs = fig.add_gridspec(5, 2)
    for i in range(len(mu)):
        f_ax1 = fig.add_subplot(gs[i%5,int(i>=5)])
        plt.imshow(mu[i].reshape(16,16), cmap = 'gray')

def assignment_9():
    # Load data
    cwd = os.getcwd()
    file_name = 'usps.mat'
    path_to_data = cwd + '/data/' + file_name
    assert os.path.exists(path_to_data), "The path to the data does not exist."
    data = sio.loadmat(path_to_data)
    data_labels = data['data_labels']
    usps_data = data['data_patterns'].T
    kn = 10
    
    # k-means on 5 usps data sets
    mu, r, loss = kmeans(X = usps_data, k = kn)
    print('Centroids: ' + str(mu))
    centroid_visualization(mu, title = 'k-means')
    
    # GMM on 5 gaussian data sets - without k-means initialization
    pi, mu, list_sigma, likelihoods = em_gmm(X = usps_data[:1000], k = kn, tol = 0.05, max_iter = 40)
    print('Centroids: ' + str(mu))
    centroid_visualization(mu, title = 'GMM')
    
    # GMM on 5 gaussian data sets - with k-means initialization
    pi, mu, list_sigma, likelihoods = em_gmm(X = usps_data[:1000], k = kn,init_kmeans = True, tol=0.05, max_iter = 10)
    print('Centroids: ' + str(mu))
    centroid_visualization(mu, title = 'GMM_km')
    
    # dendogram of hierarchical clustering with 20 starting clusters
    kn = 20
    # k-means on 5 usps data sets
    mu, r, loss = kmeans(X = usps_data, k = kn)
    # hierarchical clustering starting from kmean results
    R, kmloss, mergeidx = kmeans_agglo(usps_data,r)
    # plot dendrogram
    agglo_dendro(kmloss, mergeidx)
    
    # Visualization of the centroids found by the hierarchical agglomerative clustering for every iteration step
    # Merged clusters of the previous iteration is the last element of every column
    fig = plt.figure(figsize=(16, 16))
    gs = fig.add_gridspec(len(R)+1, len(R))
    for i, r in enumerate(R):
        # calculate init centroids
        w_q = np.array([np.sum(usps_data[r == j], axis = 0) / (np.sum(r == j)) for j in np.unique(r)])
        for j, mu in enumerate(w_q):
            f_ax1 = fig.add_subplot(gs[j,i])
            plt.imshow(mu.reshape(16,16), cmap = 'gray')
            if (i > 0 and j == 0):
                plt.title('iter: {}'.format(i))


#Assignment 10

"""
ASSIGNMENT 10: a few functions were edited for the purposes of this assignment. 
"""


# CHANGE POS LABEL TO -1 FOR THIS ASSIGNMENT
def auc(y_true, y_pred, plot=False):
    # 1. FIND ROC CURVE POINTS & FPR/TPR
    pos_label = -1
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


# edited version that returns likelihoods, not loglik
def em_gmm(X, k, max_iter=100, init_kmeans=False, tol=0.00001, converge_tol=0.0001):
    """
    This function applies the EM algorithm for Gaussian Mixture Models.

    Inputs:
    X = data (nxd)
    k = number of gaussian components
    max_iter = the maximum amount of iterations attempted to find convergence
    init_kmeans = Initialises the EM algorithm using kmeans function, if True. Default is False.
    tol = The tolerance set for the convergence condition
    converge_tol = Tolerance for the convergence condition (optional)

    Outputs:
    pi = probability that a datapoint belongs to a cluster (1xk)
    mu = center points of clusters (kxd)
    sigma = list of k dxd covariance matrices
    loglik = the loglikehlihood at each iteration
    """

    if init_kmeans == True:
        # 1.a INIT_KMEANS
        mu, r, _ = kmeans(X, k)
        unique, counts = np.unique(r, return_counts=True)
        pi = counts / np.sum(counts)

    else:
        # 1.b RANDOM INITIALISATIONS
        pi = np.full(shape=(k, 1), fill_value=1 / k)  # kx1
        rand_samples = np.random.choice(X.shape[0], size=(k,), replace=False)  # choose k random data points
        mu = X[rand_samples]  # centroid initialisation as random points, kxd

    # setup storage and loop
    sigma = [np.eye(X.shape[1]) for i in range(k)]  # dxd
    likelihoods = np.zeros(shape=(X.shape[0], k), dtype=float)  # nxk
    converged = False
    iteration = 1
    while (not converged) & (iteration <= max_iter):

        print('Iteration Number:\n', iteration)

        # 2. E-STEP - compute new likelihoods and responsibilities
        old_likelihoods = copy.deepcopy(likelihoods)
        # print('Old likelihoods\n', old_likelihoods)

        # 2.1 first find all k likelihoods
        for i in range(k):
            # nx1                             1x1 X nx1  = nx1
            likelihood = (pi[i] * norm_pdf(X, mu[i], sigma[i]))  # norm_pdf written to handle mu=(1xd) only
            likelihoods.T[i] = likelihood

        # CALC LOGLIK
        loglik = np.log(np.sum(likelihoods, axis=1)).sum()
        print('Loglikelihood\n', loglik)

        # 2.2 use likelihoods to calculate individual k responsibilities
        # nxk            nxk              nx1
        responsibilities = likelihoods / np.sum(likelihoods, axis=1).reshape(likelihoods.shape[0], 1)

        # 3. M-STEP - compute new n,pi,mu,sigma
        # 1xk
        n = np.sum(responsibilities, axis=0)
        # 1xk
        pi = n / np.sum(n, axis=0)
        # kxd                    (nxkx0)x(nx0xd)=nxkxd --> kxd / kx1
        mu = np.sum(responsibilities[:, :, None] * X[:, None, :], axis=0) / n.reshape(n.shape[0], 1)
        # kxdxd         =  sum ((nxkx0x0)     x    (nxkxdx0)x(nxkx0xd)) = nxkxdxd-->kxdxd/kx0x0
        sigma = np.sum(responsibilities[:, :, None, None] * (X[:, None, :, None] - mu[None, :, :, None]) * (
                    X[:, None, None, :] - mu[None, :, None, :]), axis=0) / n[:, None, None]
        #   (nx0xdx0-nxkx0x0)-->(nxkxdx0)
        # add regularisation term, tol
        sigma = sigma + tol * np.eye(X.shape[1])

        # break condition - only runs from second iteration to prevent log of old_likelihoods, which is 0 in iteration 1
        if iteration > 1:
            if (np.log(np.sum(old_likelihoods, axis=1)).sum() - loglik).all() < converge_tol:
                converged = True

        iteration = iteration + 1

    # return as a list of covariances
    list_sigma = [sigma[i, :, :] for i in range(k)]
    return pi, mu, list_sigma, likelihoods


# compare both methods and plot on same axis y_pred1:gammaidx y_pred2:em_gmm
def auc_compare(y_true, y_pred1, y_pred2, plot=False):
    # FOR Y_PRED1
    # 1. FIND ROC CURVE POINTS & FPR/TPR
    pos_label = -1
    y_true1 = (y_true == pos_label)  # boolean vec of true labels

    # arrange predictions in descending order (indexes)
    descending_scores = np.argsort(y_pred1, kind='mergesort')[::-1]
    # ascending_scores=np.argsort(y_pred,kind='mergesort')[::1]
    y_pred1 = y_pred1[descending_scores]
    y_true1 = y_true1[descending_scores]

    # determine distinct values to create an index of decreasing values
    # 'predicted value in y_pred where lower values tend to correspond to label -1 and higher values to label +1'
    distinct_values_idx = np.where(np.diff(y_pred1))[0]  # length n-1 as calculating differences
    distinct_descending_scores_idx = np.r_[distinct_values_idx, y_true.size - 1]  # add last entry

    tps = np.cumsum(y_true1)[distinct_descending_scores_idx]  # cumulative sum of true positives using idx
    fps = 1 - tps + distinct_descending_scores_idx  # same as cum sum of false positives

    # add 0,0 position for ROC curve
    tps = np.r_[0, tps]
    fps = np.r_[0, fps]

    # false/true positive rate
    fpr = fps / fps[-1]  # rate=sum/max
    tpr = tps / tps[-1]

    # 2.PLOT ROC CURVE POINTS
    if plot == True:
        fig, ax = plt.subplots()
        ax.plot(fpr, tpr, label='Gamma Idx Algorithm')

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
    area1 = end * np.trapz(tpr, fpr)

    # FOR Y_PRED2
    # 1. FIND ROC CURVE POINTS & FPR/TPR
    pos_label = -1
    y_true2 = (y_true == pos_label)  # boolean vec of true labels

    # arrange predictions in descending order (indexes)
    descending_scores = np.argsort(y_pred2, kind='mergesort')[::-1]
    # ascending_scores=np.argsort(y_pred,kind='mergesort')[::1]
    y_pred2 = y_pred2[descending_scores]
    y_true2 = y_true2[descending_scores]

    # determine distinct values to create an index of decreasing values
    # 'predicted value in y_pred where lower values tend to correspond to label -1 and higher values to label +1'
    distinct_values_idx = np.where(np.diff(y_pred2))[0]  # length n-1 as calculating differences
    distinct_descending_scores_idx = np.r_[distinct_values_idx, y_true.size - 1]  # add last entry

    tps = np.cumsum(y_true2)[distinct_descending_scores_idx]  # cumulative sum of true positives using idx
    fps = 1 - tps + distinct_descending_scores_idx  # same as cum sum of false positives

    # add 0,0 position for ROC curve
    tps = np.r_[0, tps]
    fps = np.r_[0, fps]

    # false/true positive rate
    fpr = fps / fps[-1]  # rate=sum/max
    tpr = tps / tps[-1]

    # 2.PLOT ROC CURVE POINTS
    if plot == True:
        ax.plot(fpr, tpr, label='EM GMM Algorithm')
        ax.plot(np.arange(0, 1.1, 0.1), np.arange(0, 1.1, 0.1), label='Random guesses')
        ax.set_ylabel('True Positive Rate (TPR)')
        ax.set_xlabel('False Positive Rate (FPR)')
        ax.set_title('ROC Curve')
        ax.legend()

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
    area2 = end * np.trapz(tpr, fpr)

    return area1, area2


# ====================================================================================

# this performs all analyses for assignment 10
def assignment_10():
    # load data
    lab_data = np.load('lab_data.npz')
    lst = lab_data.files
    for item in lst:
        print(item)
        print(lab_data[item])

    Y = lab_data['Y']
    X = lab_data['X']

    column_names = ['X0', 'X1', 'X2', 'Y']
    df = pd.DataFrame(columns=column_names)
    df['X0'] = X.T[0]
    df['X1'] = X.T[1]
    df['X2'] = X.T[2]
    df['Y'] = Y

    # visualise data
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(xs=df[df['Y'] == 1]['X0'], ys=df[df['Y'] == 1]['X1'], zs=df[df['Y'] == 1]['X2'], marker='x', s=4, c='y',
               label='inliers')
    ax.scatter(xs=df[df['Y'] == -1]['X0'], ys=df[df['Y'] == -1]['X1'], zs=df[df['Y'] == -1]['X2'], marker='x', s=4,
               c='b', label='outliers')
    ax.set_title('Plot of lab_data.npz', )
    ax.legend()
    plt.show()

    # apply em_gmm 30 times to lab_data --> incase of local optimum
    fig = plt.figure(figsize=(18, 120))
    gs = fig.add_gridspec(15, 2)
    all_loglik = []
    all_mu = []
    for i in range(30):
        pi, mu, sigma, loglik = em_gmm(X, k=3, init_kmeans=True, converge_tol=0.001)
        all_loglik.append(loglik)
        all_mu.append(mu)
        print('mu', i, '\n', mu)
        print('loglik', i, '\n', loglik)
        f_ax1 = fig.add_subplot(gs[i % 15, int(i >= 15)], projection='3d')
        f_ax1.scatter(xs=df[df['Y'] == 1]['X0'], ys=df[df['Y'] == 1]['X1'], zs=df[df['Y'] == 1]['X2'], marker='x', s=4,
                      c='y', label='inliers')
        f_ax1.scatter(xs=df[df['Y'] == -1]['X0'], ys=df[df['Y'] == -1]['X1'], zs=df[df['Y'] == -1]['X2'], marker='x',
                      s=4, c='b', label='outliers')
        f_ax1.scatter(xs=mu.T[0], ys=mu.T[1], zs=mu.T[2], marker='x', s=50, c='r', depthshade=False,
                      label='cluster centers')
        f_ax1.set_xlabel('X0')
        f_ax1.set_ylabel('X1')
        f_ax1.set_zlabel('X2')
        f_ax1.legend()
    # report mean loglik and mu by selecting results from a typical solution

    # apply gammaidx with neighbours from 2 to 30 and see what works best
    all_gamma_idx = []
    all_auc = []
    for i in range(2, 30):
        gamma = gammaidx(X=X, k=i)
        auc_ = auc(Y,
                   gamma)
        all_gamma_idx.append(gamma)
        all_auc.append(auc_)

    # best performance is k=23 and auc is 0.878 for gammaidx
    print(all_auc[np.argmax(all_auc)])
    print('k\n', np.argmax(all_auc) + 2)
    gamma = gammaidx(X=X, k=np.argmax(all_auc) + 2)

    # apply em_gmm with different tolerances to see what works best
    all_scores = []
    all_auc = []
    tolerances = [0.00001, 0.00005, 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5]
    for i in tolerances:
        pi, mu, list_sigma, likelihoods = em_gmm(X=X, k=3, init_kmeans=True, tol=i)
        scores = 1 / np.sum(likelihoods, axis=1)
        auc_ = auc(Y,scores)
        all_scores.append(scores)
        all_auc.append(auc_)

    # results
    print("Best tolerance\n", tolerances[np.argmax(all_auc)])
    print("Best AUC\n", all_auc[np.argmax(all_auc)])

    # comparative plot
    area1, area2 = auc_compare(Y, gamma, all_scores[np.argmax(all_auc)], plot=True)

    print("Gamma AUC",area1)
    print("GMM AUC",area2)
