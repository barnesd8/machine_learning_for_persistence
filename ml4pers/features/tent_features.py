def tent_features(X_train, X_test, d=5, padding=.05):
    """This function computes a training and testing set of features based on tent functions
    
    Args:
       train (array):  training set
       test (array):  testing set
       d (double):  parameter, default is 5.
       padding (double): parameter, default is .05.

    Returns:
       (array): X_train_features, features from the training set
       (array): X_test_features, features from the testing set

    """

    import numpy as np
    import ATS
    import time

    centers, delta = ATS.box_centers(X_train, d, padding) 

    start = time.time()

    X_train_features = ATS.get_all_features_boxes(X_train, centers, delta)

    X_test_features = ATS.get_all_features_boxes(X_test, centers, delta)

    end = time.time()
    print('Computing features took (seconds):{}'.format(end-start))
    return X_train_features, X_test_features
