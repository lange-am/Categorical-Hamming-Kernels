# Categorical-Hamming-Kernels

Kernel functions K(x, y) for categorical feature vectors x, y based on Hamming distance. Allows us to use kernel methods like _SVM_, _kernel regression_ or _kernel PCA_ with categorical features. The kernels reflect whether two corresponding categorical components of feature vectors are equal (Hamming distance approach). The components are also weighted by their cardinality (number of unique values).

Couto, J. (2005, September). Kernel k-means for categorical data. In International Symposium on Intelligent Data Analysis (pp. 46-56). Springer, Berlin, Heidelberg.

    Example:
        from cat_kernels import SSK_kernel, diffusion_kernel, kernel_in_parallel
        K = kernel_parallel(X, Y, SSK_kernel, {'lambd': 0.5}, n_jobs=4)     # X[n_samples_x, n_cat_features], Y[n_samples_y, n_cat_features]
      or
        K = kernel_parallel(X, Y, diffusion_kernel, {'beta': 3.0}, n_jobs=-1)
     
    =======

    Example with SVM classifier:
        # as if we precompute linear kernel: https://scikit-learn.org/stable/modules/svm.html#using-the-gram-matrix)
        from sklearn import svm
        from cat_kernels import SSK_kernel, kernel_in_parallel 

        K_train = kernel_in_parallel(X_train, X_train, SSK_kernel, {'lambd': 0.6})
        clf = svm.SVC(kernel='precomputed', C=...)
        clf.fit(K_train, y_train)

        K_test = kernel_in_parallel(X_test, X_train, SSK_kernel, {'lambd': 0.6})
        y_pred = clf.predict(K_test) # or use clf.decision_function(K_test)
    
    =======
    
    Example of tuning of SVM and kernel hyperparameters:
        from cat_kernels import SSK_kernel, kernel_in_parallel 
        from sklearn import svm
        from sklearn.model_selection import GridSearchCV

        for lambd in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.99]:
            K = cat_kernels.kernel_in_parallel(X, X, SSK_kernel, {'lambd': lambd})
            clf = GridSearchCV(svm.SVC(kernel='precomputed'),
                              {'C': [0.1, 1, 5, 8, 10, 25, 50, 75, 100, 150, 200]}, 
                              scoring='roc_auc', cv = 3, n_jobs=-1, refit=False)
            clf.fit(K, y_class)
            print('lambda: {l:.2f}, best C: {C:3}, cv score: {s}'.format(l=lambd, C=clf.best_params_['C'], s=clf.best_score_))
    Output:
        lambda: 0.10, best C:   1, cv score: 0.8297647468437667
        lambda: 0.20, best C:   1, cv score: 0.850261598295381
        lambda: 0.30, best C:   1, cv score: 0.8507940216294699
        lambda: 0.40, best C:   1, cv score: 0.8517652686395335
        lambda: 0.50, best C: 0.1, cv score: 0.855391733041002
        lambda: 0.60, best C:   1, cv score: 0.8583185887661827
        lambda: 0.70, best C:   5, cv score: 0.8603784000220034
        lambda: 0.80, best C:  25, cv score: 0.8620498096772128
        lambda: 0.85, best C:  50, cv score: 0.8626114931434566 # The best!
        lambda: 0.90, best C: 200, cv score: 0.8622268181667948
        lambda: 0.95, best C:  75, cv score: 0.8618465252945907
        lambda: 0.99, best C: 200, cv score: 0.8562467975639977
    
    
    Similarly for diffusion kernel:
        from cat_kernels import diffusion_kernel, kernel_in_parallel 
        from sklearn import svm
        from sklearn.model_selection import GridSearchCV

        for beta in [2.0, 2.5, 3.0, 3.1, 3.2, 3.3, 3.4]:
            K = kernel_in_parallel(X, X, diffusion_kernel, {'beta': beta})
            clf = GridSearchCV(svm.SVC(kernel='precomputed'),
                               {'C': [1, 5, 10, 100, 125, 150, 175, 200, 250, 300, 350, 400, 450, 500, 550, 800, 850]}, 
                               scoring='roc_auc', cv = 3, n_jobs=-1, refit=False)
            clf.fit(K, y_class)
            print('beta: {beta:.2f}, best C: {C:3}, cv score: {s}'.format(beta=beta, C=clf.best_params_['C'], s=clf.best_score_))
    Output:
        beta: 2.00, best C:   5, cv score: 0.846704317457808
        beta: 2.50, best C: 100, cv score: 0.8428354288139154
        beta: 3.00, best C: 150, cv score: 0.8506024350777297 # The best!
        beta: 3.10, best C: 250, cv score: 0.8500461330694771
        beta: 3.20, best C: 350, cv score: 0.8505631568643901
        beta: 3.30, best C: 500, cv score: 0.850370282880518
        beta: 3.40, best C: 800, cv score: 0.8503134651089529
        
    
        
     
    
    
