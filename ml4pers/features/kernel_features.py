
def kernel_features(train, test, s = .3):

    """This function computes a training and testing set of features based on the multi-scale kernel method when given a training and testing set of persistence diagrams
    
    Args:
       train (array):  training set
       test (array):  testing set
       s (double):  parameter for kernel, default is .3.

    Returns:
       (array): X_train_features, features from the training set
       (array): X_test_features, features from the testing set

    """
    from persim import heat
    import numpy as np
    import time

    n_train = len(train)
    n_test = len(test)
    X_train_features = np.zeros((n_train, n_train))
    X_test_features = np.zeros((n_test, n_train))
    
    start = time.time()
    for i in range(0,n_train):
        for j in range(0,i):
            print("train: ", j)
            dgm0 = train[i]
            dgm1 = train[j]
            hk = heat(dgm0, dgm1, sigma = s) 
            X_train_features[i,j] = hk
            X_train_features[j,i] = hk
        for j in range(0,n_test):
            print("test: ", j)
            dgm0 = train[i]
            dgm1 = test[j]
            hk = heat(dgm0, dgm1, sigma = s)        
            X_test_features[j,i] = hk

    print(time.time()-start)
    return X_train_features, X_test_features