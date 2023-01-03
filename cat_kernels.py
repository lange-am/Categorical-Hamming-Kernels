"""Hamming distance based kernels for cagorical features.

Couto, J. (2005, September). Kernel k-means for categorical data. In International Symposium on Intelligent Data Analysis (pp. 46-56). Springer, Berlin, Heidelberg.

"""

import numpy as np
import multiprocessing
from functools import partial


def SSK_kernel(X, Y, lambd=0.5, normalised=True):  
    """
        String subsequence kernel (SSK) for categorical feature data.
        For X[n_samples_x, n_features], Y[n_samples_y, n_features] returns symetrical kernel matrix of size [n_samples_x, n_samples_y].

        Usage:
            SSK_kernel(X_train, X_train, lambd),
        and
            SSK_kernel(X_test, X_train, lambd).

        lambd is in (0, 1). Numbers of unique values in categories are calculated using Y matrix (used as train set).
        If normalised == True then the normalised kernel K(x, y)/sqrt(K(x, x)*K(y, y)) is returned.   
    """

    X, Y = np.array(X), np.array(Y)
    assert(X.shape[1] == Y.shape[1])

    lambd2 = lambd * 2
    lambd_sqr = lambd * lambd

    n = Y.shape[1]
    DD = [len(np.unique(Y[:, j])) for j in range(n)]
    L2Dm1 = [lambd_sqr * (d-1) + 1 for d in DD]
    L2Dm2 = [lambd_sqr * (d-2) + lambd2 for d in DD]

    K = np.zeros([X.shape[0], Y.shape[0]])
    if X is Y:  # calculate only not above diagonal elements, for ex. in case SSK_kernel(X_train, X_train, lambd)
        for ix, x in enumerate(X):
            for iy in range(ix+1):
                delta = x != X[iy]
                K[ix, iy] = np.prod([l2dm2 if d else l2dm1 for l2dm2, l2dm1, d in zip(L2Dm2, L2Dm1, delta)])
        K = K + np.tril(K, -1).T
    else:  # common case
        for ix, x in enumerate(X):
            for iy, y in enumerate(Y):
                delta = x != y
                K[ix, iy] = np.prod([l2dm2 if d else l2dm1 for l2dm2, l2dm1, d in zip(L2Dm2, L2Dm1, delta)])
        
    if normalised:
        K = K / np.prod(L2Dm1)

    return K


def diffusion_kernel(X, Y, beta=0.5):  
    """
        String subsequence kernel (SSK) for categorical feature data.
        For X[n_samples_x, n_features], Y[n_samples_y, n_features] returns symetrical kernel matrix of size [n_samples_x, n_samples_y].

        Usage:
            SSK_kernel(X_train, X_train),
        and
            SSK_kernel(X_test, X_train).

        beta is in (0, +infty), for ex. between 0.1 and 2.0. The numbers of unique values in categories are calculated using Y matrix.  
    """

    X, Y = np.array(X), np.array(Y)
    assert(X.shape[1] == Y.shape[1])

    n = Y.shape[1]
    DD = np.array([len(np.unique(Y[:, j])) for j in range(n)])
    EE = np.exp(-beta * DD)
    EE = (1 - EE) / (1 + (DD-1) * EE) 

    K = np.zeros([X.shape[0], Y.shape[0]])
    if X is Y:
        for ix, x in enumerate(X):
            for iy in range(ix+1):
                delta = x != X[iy]
                r = []
                for ee, d in zip(EE, delta):
                    if d:
                        r.append(ee)
                K[ix, iy] = np.prod(r)
        K = K + np.tril(K, -1).T
    else:
        for ix, x in enumerate(X):
            for iy, y in enumerate(Y):
                delta = x != y
                r = []
                for ee, d in zip(EE, delta):
                    if d:
                        r.append(ee)
                K[ix, iy] = np.prod(r)
    return K


def kernel_in_parallel(X, Y, kernel_func, kernel_params, n_jobs=-1):
    """
    Utility for the parallelization of the kernel calculation across groups of rows in X-matrix.

    Example:
        K = kernel_parallel(X, Y, SSK_kernel, {'lambd': 0.5}, n_jobs=4) 

    Example with SVM classifier:
        # as if we precompute linear kernel: https://scikit-learn.org/stable/modules/svm.html#using-the-gram-matrix)
        from sklearn import svm
        from cat_kernels import SSK_kernel, kernel_in_parallel 

        K_train = kernel_in_parallel(X_train, X_train, SSK_kernel, {'lambd': 0.6})
        clf = svm.SVC(kernel='precomputed', C=...)
        clf.fit(K_train, y_train)

        K_test = kernel_in_parallel(X_test, X_train, SSK_kernel, {'lambd': 0.6})
        y_pred = clf.predict(K_test)
    """

    if n_jobs == -1:
        CPU_NUM = multiprocessing.cpu_count()
    else: 
        CPU_NUM = n_jobs
    CPU_NUM = min(X.shape[0], CPU_NUM)

    with multiprocessing.Pool(CPU_NUM) as p:
        r = p.starmap(
            partial(kernel_func, **kernel_params), 
            [(X[i::CPU_NUM], Y) for i in range(CPU_NUM)])

    K = np.zeros([X.shape[0], Y.shape[0]])
    for i in range(CPU_NUM):
        K[i::CPU_NUM, :] = r[i]

    return K
