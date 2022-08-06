from persim import heat
import numpy as np
import ATS
import time
from sklearn import mixture
import multidim
from multidim.covertree import CoverTree
from multidim.models import CDER
import hdbscan
from gudhi.representations.vector_methods import Landscape
from persim import PersImage
from sklearn.neighbors import DistanceMetric

def kernel_features(train, test, s):
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

def tent_features(X_train, X_test, d=5, padding=.05):
    centers, delta = ATS.box_centers(X_train, d, padding) 

    start = time.time()

    X_train_features = ATS.get_all_features_boxes(X_train, centers, delta)

    X_test_features = ATS.get_all_features_boxes(X_test, centers, delta)

    end = time.time()
    print('Computing features took (seconds):{}'.format(end-start))
    return X_train_features, X_test_features

def adaptive_features(X_train, X_test, model, y_train, d=25):
    if model == "gmm":
        print('Begin GMM...')
        start = time.time()
        X_train_temp = np.vstack(X_train)
        gmm_f_train=[]
        for i in range(len(X_train)):
            gmm_f_train.append(y_train[i]*np.ones(len(X_train[i])))
        gmm_f_train = np.concatenate(gmm_f_train)

        gmm = mixture.BayesianGaussianMixture(n_components=d, covariance_type='full', max_iter=int(10e4)).fit(X_train_temp, gmm_f_train)

        ellipses = []
        for i in range(len(gmm.means_)):
            L, v = np.linalg.eig(gmm.covariances_[i])
            temp = {'mean':gmm.means_[i], 'std':np.sqrt(L), 'rotation':v.transpose(), 'radius':max(np.sqrt(L)), 'entropy':gmm.weights_[i]}
            temp['std'] = 3*temp['std']
            ellipses.append(temp)

        X_train_features = ATS.get_all_features(X_train, ellipses, ATS.f_ellipse)

        X_test_features = ATS.get_all_features(X_test, ellipses, ATS.f_ellipse)

        end = time.time()
        print('Computing gmm features took (seconds):{}'.format(end-start))
        
    elif model == "hdb":
        print('Begin HDBSCAN...')
        start = time.time()
        X_train_temp = np.vstack(X_train)

        clusterer = hdbscan.HDBSCAN()

        clusterer.fit(X_train_temp)

        num_clusters = clusterer.labels_.max()

        ellipses = []
        for i in range(num_clusters):
            cluster_i = X_train_temp[clusterer.labels_ == i]

            en = np.mean(clusterer.probabilities_[clusterer.labels_ == i])

            mean = np.mean(cluster_i, axis=0)
            cov_matrix = np.cov(cluster_i.transpose())

            L,v = np.linalg.eig(cov_matrix)

            temp = {'mean':mean, 'std':np.sqrt(L), 'rotation':v.transpose(), 'radius':max(np.sqrt(L)), 'entropy':en}
            temp['std'] = 2*temp['std']
            ellipses.append(temp)

        X_train_features = ATS.get_all_features(X_train, ellipses, ATS.f_ellipse)

        X_test_features = ATS.get_all_features(X_test, ellipses, ATS.f_ellipse)

        end = time.time()
        print('Computing hdbscan features took (seconds):{}'.format(end-start))
        
    elif model == "cder":

        y_train_cder = y_train.copy()

        print('Begin CDER...')
        start = time.time()

        pc_train = multidim.PointCloud.from_multisample_multilabel(X_train, y_train_cder)
        ct_train = CoverTree(pc_train)

        cder = CDER(parsimonious=True)

        cder.fit(ct_train)

        cder_result = cder.gaussians

        ellipses = []
        for c in cder_result:
            temp = {key:c[key] for key in ['mean', 'std', 'rotation', 'radius', 'entropy']}
            temp['std'] = 3*temp['std']
            ellipses.append(temp)

        X_train_features = ATS.get_all_features(X_train, ellipses, ATS.f_ellipse)

        X_test_features = ATS.get_all_features(X_test, ellipses, ATS.f_ellipse)

        end = time.time()
        print('Computing features from H_1 took (seconds):{}'.format(end-start))
        
    else:
        print("Not a valid model type")
    return X_train_features, X_test_features

def carlsson_coordinates(X_train, X_test):
    n = len(X_train)
    X_train_features_cc1 = np.zeros(n)
    X_train_features_cc2 = np.zeros(n)
    X_train_features_cc3 = np.zeros(n)
    X_train_features_cc4 = np.zeros(n)
    start = time.time()
    ymax = 0
    for i in range(0,n):
        if len(X_train[i])>0:
            b = np.max(X_train[i][:,1])
        else:
            b = ymax
        if ymax < b:
            ymax = b
        else:
            ymax = ymax
    print(ymax)
    for i in range(0,n):
        if len(X_train[i])>0:
            x = X_train[i][:,0]
            y = X_train[i][:,1]
            X_train_features_cc1[i] = sum(x*(y-x))
            X_train_features_cc2[i] = sum((ymax - y)*(y-x))
            X_train_features_cc3[i] = sum(x**2*(y-x)**4)
            X_train_features_cc4[i] = sum((ymax-y)**2*(y-x)**4)
        else:
            X_train_features_cc1[i] = 0
            X_train_features_cc2[i] = 0
            X_train_features_cc3[i] = 0
            X_train_features_cc4[i] = 0

    n = len(X_test)
    X_test_features_cc1 = np.zeros(n)
    X_test_features_cc2 = np.zeros(n)
    X_test_features_cc3 = np.zeros(n)
    X_test_features_cc4 = np.zeros(n)
    ymax = 0
    for i in range(0,n):
        if len(X_test[i])>0:
            b = np.max(X_test[i][:,1])
        else:
            b = ymax
        if ymax < b:
            ymax = b
        else:
            ymax = ymax
    for i in range(0,n):
        if len(X_test[i])>0:
            x = X_test[i][:,0]
            y = X_test[i][:,1]
            X_test_features_cc1[i] = sum(x*(y-x))
            X_test_features_cc2[i] = sum((ymax - y)*(y-x))
            X_test_features_cc3[i] = sum(x**2*(y-x)**4)
            X_test_features_cc4[i] = sum((ymax-y)**2*(y-x)**4)
        else:
            X_test_features_cc1[i] = 0
            X_test_features_cc2[i] = 0
            X_test_features_cc3[i] = 0
            X_test_features_cc4[i] = 0
    print("Total Time (Carlsson Coordinates): ", time.time()-start)
    return X_train_features_cc1, X_train_features_cc2, X_train_features_cc3, X_train_features_cc4, X_test_features_cc1, X_test_features_cc2, X_test_features_cc3, X_test_features_cc4

def landscape_features(X_train, X_test, num_landscapes=5, resolution=100):
    start = time.time()
    landscapes = Landscape(num_landscapes, resolution)
    lr = landscapes.fit(X_train)
    X_train_features = lr.transform(X_train)
    X_test_features = lr.transform(X_test)
    print("Total Time (Landscape Features): ", time.time()-start)
    return X_train_features, X_test_features

def persistence_image_features(X_train, X_test, pixels=[20,20], spread=1):
    start = time.time()
    pim = PersImage(pixels=pixels, spread=spread)
    imgs_train = pim.transform(X_train)
    X_train_features = np.array([img.flatten() for img in imgs_train])
    pim = PersImage(pixels=pixels, spread=spread)
    imgs_test = pim.transform(X_test)
    X_test_features = np.array([img.flatten() for img in imgs_test])
    print("Total Time (Persistence Images): ", time.time()-start)
    return X_train_features, X_test_features

def fast_hk(dgm0,dgm1,sigma=.4):
    dist = DistanceMetric.get_metric('euclidean')
    dist1 = (dist.pairwise(dgm0,dgm1))**2
    Qc = dgm1[:,1::-1]
    dist2 = (dist.pairwise(dgm0,Qc))**2
    exp_dist1 = sum(sum(np.exp(-dist1/(8*sigma))))
    exp_dist2 = sum(sum(np.exp(-dist2/(8*sigma))))
    hk = (exp_dist1-exp_dist2)/(8*np.pi*sigma)
    return(hk)

def heat_kernel_approx(dgm0, dgm1, sigma=.4):
    return np.sqrt(fast_hk(dgm0, dgm0, sigma) + fast_hk(dgm1, dgm1, sigma) - 2*fast_hk(dgm0, dgm1, sigma))

def fast_kernel_features(train, test, s):
    n_train = len(train)
    n_test = len(test)
    X_train_features = np.zeros((n_train, n_train))
    X_test_features = np.zeros((n_test, n_train))
    
    start = time.time()
    for i in range(0,n_train):
        if i % 5 == 0:
            print("Iteration: ", i)
            print("Total Time: ", time.time() - start)
            print("Iterations left: " , n_train - i)
        for j in range(0,i):
            dgm0 = train[i]
            dgm1 = train[j]
            hka = heat_kernel_approx(dgm0, dgm1, sigma = s) 
            X_train_features[i,j] = hka
            X_train_features[j,i] = hka
        for j in range(0,n_test):
            dgm0 = train[i]
            dgm1 = test[j]
            hka = heat_kernel_approx(dgm0, dgm1, sigma = s)        
            X_test_features[j,i] = hka

    print("Total Time (Kernel): ", time.time()-start)
    return X_train_features, X_test_features

    
