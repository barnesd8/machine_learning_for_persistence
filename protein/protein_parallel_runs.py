import numpy as np
import itertools
import os
import hdbscan
import sys
import time
import pandas as pd
import itertools
import pickle
import warnings

from copy import deepcopy
from ripser import ripser
from persim import plot_diagrams
from matplotlib.patches import Ellipse
from numba import jit, njit, prange
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import RidgeClassifier, LogisticRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.svm import LinearSVC, SVC, NuSVC
from sklearn.kernel_ridge import KernelRidge
from sklearn import mixture

import matplotlib.pyplot as plt

from gudhi.representations.vector_methods import Landscape

from persim import PersImage
from persim import heat

sys.path.append('../persistence_methods')
import ATS
from persistence_methods import kernel_features
from persistence_methods import tent_features
from persistence_methods import carlsson_coordinates
from persistence_methods import adaptive_features
from persistence_methods import landscape_features
from persistence_methods import persistence_image_features
from persistence_methods import fast_kernel_features

from mpi4py import MPI

warnings.simplefilter("ignore")
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

Data = pickle.load(open("persistence/diagrams.pickle", "rb"))
X_dgm0 = [d['h0'] for d in Data]
X_dgm1 = [d['h1'] for d in Data]

# We need to perturbate H_0 to use CDER.
for h0 in X_dgm0:
    h0[:,1][h0[:,1]==np.inf] = 2 # Changge all inf values in H_0 for 10.

# Determine Index
if rank == 0:
    index = np.arange(start = 1,stop = 56)
    print(index)
    index = np.array_split(index, size)
else:
    index = None

index = comm.scatter(index)
print(index)
nruns = len(index)

tent_train_accuracy_ridge = np.zeros(nruns)
tent_test_accuracy_ridge = np.zeros(nruns)

tent_train_accuracy_svm= np.zeros(nruns)
tent_test_accuracy_svm = np.zeros(nruns)

gmm_train_accuracy_ridge = np.zeros(nruns)
gmm_test_accuracy_ridge = np.zeros(nruns)

gmm_train_accuracy_svm = np.zeros(nruns)
gmm_test_accuracy_svm = np.zeros(nruns)

images_train_accuracy_ridge = np.zeros(nruns)
images_test_accuracy_ridge = np.zeros(nruns)

images_train_accuracy_svm = np.zeros(nruns)
images_test_accuracy_svm = np.zeros(nruns)

landscapes_train_accuracy_ridge = np.zeros(nruns)
landscapes_test_accuracy_ridge = np.zeros(nruns)

landscapes_train_accuracy_svm = np.zeros(nruns)
landscapes_test_accuracy_svm = np.zeros(nruns)

kernel_train_accuracy_svm = np.zeros(nruns)
kernel_test_accuracy_svm = np.zeros(nruns)

carlson_train_accuracy_ridge = np.zeros(nruns)
carlson_test_accuracy_ridge = np.zeros(nruns)

carlson_train_accuracy_svm = np.zeros(nruns)
carlson_test_accuracy_svm = np.zeros(nruns)

start = MPI.Wtime()
for t in index:
    k = 0
    print("Starting Index ", t)
    train_index_name = 'data/Index/TrainIndex' + str(t) + '.npy'
    test_index_name = 'data/Index/TestIndex' + str(t) + '.npy'
    y_train_name = 'data/Index/TrainLabel' + str(t) + '.npy'
    y_test_name = 'data/Index/TestLabel' + str(t) + '.npy'
    train_index = np.load(train_index_name)
    train_index = train_index
    test_index = np.load(test_index_name)
    test_index = test_index
    X_dgm0_train = np.array(X_dgm0, dtype=object)[train_index]
    X_dgm0_test = np.array(X_dgm0, dtype=object)[test_index]
    X_dgm1_train = np.array(X_dgm1, dtype=object)[train_index]
    X_dgm1_test = np.array(X_dgm1, dtype=object)[test_index]
    y_train = np.load(y_train_name)
    y_train = y_train
    y_test = np.load(y_test_name)
    y_test = y_test
   

    ### Tent Features
    d = 5
    p = 1
    X_train_features_1_tent, X_test_features_1_tent = tent_features(X_dgm1_train, X_dgm1_test, d, p)
    X_train_features_0_tent, X_test_features_0_tent = tent_features(X_dgm0_train, X_dgm0_test, d, p)
    X_train_features = np.column_stack((X_train_features_0_tent, X_train_features_1_tent))
    X_test_features = np.column_stack((X_test_features_0_tent, X_test_features_1_tent))

    ### Ridge Model
    ridge_model = RidgeClassifier().fit(X_train_features, y_train)
    tent_train_accuracy_ridge[k] = ridge_model.score(X_train_features, y_train)
    tent_test_accuracy_ridge[k] = ridge_model.score(X_test_features, y_test)

    ### SVM Model
    c = 1
    svm_model = SVC(kernel='rbf', C = c).fit(X_train_features, y_train)
    tent_train_accuracy_svm[k] = svm_model.score(X_train_features, y_train)
    tent_test_accuracy_svm[k] = svm_model.score(X_test_features, y_test)

    ### Adaptive Features
    X_train_features_1_gmm, X_test_features_1_gmm = adaptive_features(X_dgm1_train, X_dgm1_test, "cder", y_train)
    X_train_features_0_gmm, X_test_features_0_gmm = adaptive_features(X_dgm0_train, X_dgm0_test, "cder", y_train)
    X_train_features = np.column_stack((X_train_features_0_gmm, X_train_features_1_gmm))
    X_test_features = np.column_stack((X_test_features_0_gmm, X_test_features_1_gmm))

    ridge_model = RidgeClassifier().fit(X_train_features, y_train)
    gmm_train_accuracy_ridge[k] = ridge_model.score(X_train_features, y_train)
    gmm_test_accuracy_ridge[k] = ridge_model.score(X_test_features, y_test)

    c = 1
    svm_model = SVC(kernel='rbf', C=c).fit(X_train_features, y_train)
    gmm_train_accuracy_svm[k] = svm_model.score(X_train_features, y_train)
    gmm_test_accuracy_svm[k] = svm_model.score(X_test_features, y_test)

    ### Persistence images
    pixels = [20,20]
    spread = 1
    X_train_features_1_imgs, X_test_features_1_imgs = persistence_image_features(X_dgm1_train, X_dgm1_test,pixels = pixels, spread = 1)
    X_train_features_0_imgs, X_test_features_0_imgs = persistence_image_features(X_dgm0_train, X_dgm0_test,pixels = pixels, spread = 1)
    X_train_features = np.column_stack((X_train_features_1_imgs,X_train_features_0_imgs))
    X_test_features = np.column_stack((X_test_features_1_imgs,X_test_features_0_imgs))

    ridge_model = RidgeClassifier().fit(X_train_features, y_train)
    images_train_accuracy_ridge[k] = ridge_model.score(X_train_features, y_train)
    images_test_accuracy_ridge[k] = ridge_model.score(X_test_features, y_test)

    c = 1
    svm_model = SVC(kernel='rbf', C = c).fit(X_train_features, y_train)
    images_train_accuracy_svm[k] = svm_model.score(X_train_features, y_train)
    images_test_accuracy_svm[k] = svm_model.score(X_test_features, y_test)

    ### Kernel Features
    s = .4
    X_train_features_1_kernel, X_test_features_1_kernel = fast_kernel_features(X_dgm1_train, X_dgm1_test, s)
    X_train_features_0_kernel, X_test_features_0_kernel = fast_kernel_features(X_dgm0_train, X_dgm0_test, s)
    X_train_features = X_train_features_1_kernel + X_train_features_0_kernel
    X_test_features = X_test_features_1_kernel + X_test_features_0_kernel
    svm_model = SVC(kernel='precomputed').fit(X_train_features, y_train)
    kernel_train_accuracy_svm[k] = svm_model.score(X_train_features, y_train)
    kernel_test_accuracy_svm[k] = svm_model.score(X_test_features, y_test)

    ### Landscape Features
    n = 5
    r = 100
    X_train_features_1_landscapes, X_test_features_1_landscapes = landscape_features(X_dgm1_train, X_dgm1_test, num_landscapes=n, resolution=r)
    X_train_features_0_landscapes, X_test_features_0_landscapes = landscape_features(X_dgm0_train, X_dgm0_test, num_landscapes=n, resolution=r)
    X_train_features = np.column_stack((X_train_features_0_landscapes, X_train_features_1_landscapes))
    X_test_features = np.column_stack((X_test_features_0_landscapes, X_test_features_1_landscapes))

    ridge_model = RidgeClassifier().fit(X_train_features, y_train)
    landscapes_train_accuracy_ridge[k] = ridge_model.score(X_train_features, y_train)
    landscapes_test_accuracy_ridge[k] = ridge_model.score(X_test_features, y_test)

    c = 1
    svm_model = SVC(kernel='rbf', C=c).fit(X_train_features, y_train)
    landscapes_train_accuracy_svm[k] = svm_model.score(X_train_features, y_train)
    landscapes_test_accuracy_svm[k] = svm_model.score(X_test_features, y_test)

    ### Carlson Coordinates
    c = 1
    X_train_features1_cc1, X_train_features1_cc2, X_train_features1_cc3, X_train_features1_cc4, X_test_features1_cc1, X_test_features1_cc2, X_test_features1_cc3, X_test_features1_cc4 = carlsson_coordinates(X_dgm1_train, X_dgm1_test)
    X_train_features0_cc1, X_train_features0_cc2, X_train_features0_cc3, X_train_features0_cc4, X_test_features0_cc1, X_test_features0_cc2, X_test_features0_cc3, X_test_features0_cc4 = carlsson_coordinates(X_dgm0_train, X_dgm0_test)
    X_train_features = np.column_stack((X_train_features1_cc1, X_train_features1_cc2, X_train_features1_cc3, X_train_features1_cc4, X_train_features0_cc1, X_train_features0_cc2, X_train_features0_cc3, X_train_features0_cc4))
    X_test_features = np.column_stack((X_test_features1_cc1,X_test_features1_cc2, X_test_features1_cc3,X_test_features1_cc4, X_test_features0_cc1,X_test_features0_cc2, X_test_features0_cc3,X_test_features0_cc4))

    ridge_model = RidgeClassifier().fit(X_train_features, y_train)
    carlson_train_accuracy_ridge[k] = ridge_model.score(X_train_features, y_train)
    carlson_test_accuracy_ridge[k] = ridge_model.score(X_test_features, y_test)

    svm_model = SVC(kernel='rbf', C = c).fit(X_train_features, y_train)
    carlson_train_accuracy_svm[k] = svm_model.score(X_train_features, y_train)
    carlson_test_accuracy_svm[k] = svm_model.score(X_test_features, y_test)

    k +=1

tent_train_accuracy_ridge = comm.gather(tent_train_accuracy_ridge, root=0)
tent_test_accuracy_ridge = comm.gather(tent_test_accuracy_ridge, root=0)

tent_train_accuracy_svm = comm.gather(tent_train_accuracy_svm, root=0)
tent_test_accuracy_svm = comm.gather(tent_test_accuracy_svm, root=0)

gmm_train_accuracy_ridge = comm.gather(gmm_train_accuracy_ridge, root=0)
gmm_test_accuracy_ridge = comm.gather(gmm_test_accuracy_ridge, root=0)

gmm_train_accuracy_svm = comm.gather(gmm_train_accuracy_svm, root=0)
gmm_test_accuracy_svm = comm.gather(gmm_test_accuracy_svm, root=0)

images_train_accuracy_ridge = comm.gather(images_train_accuracy_ridge, root=0)
images_test_accuracy_ridge = comm.gather(images_test_accuracy_ridge, root=0)

images_train_accuracy_svm = comm.gather(images_train_accuracy_svm, root=0)
images_test_accuracy_svm = comm.gather(images_test_accuracy_svm, root=0)

landscapes_train_accuracy_ridge = comm.gather(landscapes_train_accuracy_ridge, root=0)
landscapes_test_accuracy_ridge = comm.gather(landscapes_test_accuracy_ridge, root=0)

landscapes_train_accuracy_svm = comm.gather(landscapes_train_accuracy_svm, root=0)
landscapes_test_accuracy_svm = comm.gather(landscapes_test_accuracy_svm, root=0)

kernel_train_accuracy_svm = comm.gather(kernel_train_accuracy_svm, root=0)
kernel_test_accuracy_svm = comm.gather(kernel_test_accuracy_svm, root=0)

carlson_train_accuracy_ridge = comm.gather(carlson_train_accuracy_ridge, root=0)
carlson_test_accuracy_ridge = comm.gather(carlson_test_accuracy_ridge, root=0)

carlson_train_accuracy_svm = comm.gather(carlson_train_accuracy_svm, root=0)
carlson_test_accuracy_svm = comm.gather(carlson_test_accuracy_svm, root=0)

comm.Barrier()

if rank == 0:
    print("Total Time is ", MPI.Wtime()-start)

    tent_train_accuracy_ridge = np.array(tent_train_accuracy_ridge)
    tent_train_accuracy_ridge = np.reshape(tent_train_accuracy_ridge, newshape=(-1,1))
    tent_train_mean = np.mean(tent_train_accuracy_ridge)
    tent_train_std = np.std(tent_train_accuracy_ridge)

    tent_test_accuracy_ridge = np.array(tent_test_accuracy_ridge)
    tent_test_accuracy_ridge = np.reshape(tent_test_accuracy_ridge, newshape=(-1,1))
    tent_test_mean = np.mean(tent_test_accuracy_ridge)
    tent_test_std = np.std(tent_test_accuracy_ridge)

    print("Tent Features - Ridge Model")
    print("Mean Training Accuracy: ", tent_train_mean, "Mean Testing Accuracy: ", tent_test_mean, "Std Dev Training Accuracy: ", tent_train_std, "Std Dev Testing Accuracy: ", tent_test_std)

    tent_train_accuracy_svm = np.array(tent_train_accuracy_svm)
    tent_train_accuracy_svm = np.reshape(tent_train_accuracy_svm, newshape=(-1,1))
    tent_train_mean = np.mean(tent_train_accuracy_svm)
    tent_train_std = np.std(tent_train_accuracy_svm)

    tent_test_accuracy_svm = np.array(tent_test_accuracy_svm)
    tent_test_accuracy_svm = np.reshape(tent_test_accuracy_svm, newshape=(-1,1))
    tent_test_mean = np.mean(tent_test_accuracy_svm)
    tent_test_std = np.std(tent_test_accuracy_svm)

    print("Tent Features - svm Model")
    print("Mean Training Accuracy: ", tent_train_mean, "Mean Testing Accuracy: ", tent_test_mean, "Std Dev Training Accuracy: ", tent_train_std, "Std Dev Testing Accuracy: ", tent_test_std)

    gmm_train_accuracy_ridge = np.array(gmm_train_accuracy_ridge)
    gmm_train_accuracy_ridge = np.reshape(gmm_train_accuracy_ridge, newshape=(-1,1))
    gmm_train_mean = np.mean(gmm_train_accuracy_ridge)
    gmm_train_std = np.std(gmm_train_accuracy_ridge)

    gmm_test_accuracy_ridge = np.array(gmm_test_accuracy_ridge)
    gmm_test_accuracy_ridge = np.reshape(gmm_test_accuracy_ridge, newshape=(-1,1))
    gmm_test_mean = np.mean(gmm_test_accuracy_ridge)
    gmm_test_std = np.std(gmm_test_accuracy_ridge)

    print("gmm Features - Ridge Model")
    print("Mean Training Accuracy: ", gmm_train_mean, "Mean Testing Accuracy: ", gmm_test_mean, "Std Dev Training Accuracy: ", gmm_train_std, "Std Dev Testing Accuracy: ", gmm_test_std)
    
    gmm_train_accuracy_svm = np.array(gmm_train_accuracy_svm)
    gmm_train_accuracy_svm = np.reshape(gmm_train_accuracy_svm, newshape=(-1,1))
    gmm_train_mean = np.mean(gmm_train_accuracy_svm)
    gmm_train_std = np.std(gmm_train_accuracy_svm)

    gmm_test_accuracy_svm = np.array(gmm_test_accuracy_svm)
    gmm_test_accuracy_svm = np.reshape(gmm_test_accuracy_svm, newshape=(-1,1))
    gmm_test_mean = np.mean(gmm_test_accuracy_svm)
    gmm_test_std = np.std(gmm_test_accuracy_svm)

    print("gmm Features - svm Model")
    print("Mean Training Accuracy: ", gmm_train_mean, "Mean Testing Accuracy: ", gmm_test_mean, "Std Dev Training Accuracy: ", gmm_train_std, "Std Dev Testing Accuracy: ", gmm_test_std)

    images_train_accuracy_ridge = np.array(images_train_accuracy_ridge)
    images_train_accuracy_ridge = np.reshape(images_train_accuracy_ridge, newshape=(-1,1))
    images_train_mean = np.mean(images_train_accuracy_ridge)
    images_train_std = np.std(images_train_accuracy_ridge)

    images_test_accuracy_ridge = np.array(images_test_accuracy_ridge)
    images_test_accuracy_ridge = np.reshape(images_test_accuracy_ridge, newshape=(-1,1))
    images_test_mean = np.mean(images_test_accuracy_ridge)
    images_test_std = np.std(images_test_accuracy_ridge)

    print("images Features - ridge Model")
    print("Mean Training Accuracy: ", images_train_mean, "Mean Testing Accuracy: ", images_test_mean, "Std Dev Training Accuracy: ", images_train_std, "Std Dev Testing Accuracy: ", images_test_std)

    images_train_accuracy_svm = np.array(images_train_accuracy_svm)
    images_train_accuracy_svm = np.reshape(images_train_accuracy_svm, newshape=(-1,1))
    images_train_mean = np.mean(images_train_accuracy_svm)
    images_train_std = np.std(images_train_accuracy_svm)

    images_test_accuracy_svm = np.array(images_test_accuracy_svm)
    images_test_accuracy_svm = np.reshape(images_test_accuracy_svm, newshape=(-1,1))
    images_test_mean = np.mean(images_test_accuracy_svm)
    images_test_std = np.std(images_test_accuracy_svm)

    print("images Features - svm Model")
    print("Mean Training Accuracy: ", images_train_mean, "Mean Testing Accuracy: ", images_test_mean, "Std Dev Training Accuracy: ", images_train_std, "Std Dev Testing Accuracy: ", images_test_std)
    
    landscapes_train_accuracy_ridge = np.array(landscapes_train_accuracy_ridge)
    landscapes_train_accuracy_ridge = np.reshape(landscapes_train_accuracy_ridge, newshape=(-1,1))
    landscapes_train_mean = np.mean(landscapes_train_accuracy_ridge)
    landscapes_train_std = np.std(landscapes_train_accuracy_ridge)

    landscapes_test_accuracy_ridge = np.array(landscapes_test_accuracy_ridge)
    landscapes_test_accuracy_ridge = np.reshape(landscapes_test_accuracy_ridge, newshape=(-1,1))
    landscapes_test_mean = np.mean(landscapes_test_accuracy_ridge)
    landscapes_test_std = np.std(landscapes_test_accuracy_ridge)

    print("landscapes Features - ridge Model")
    print("Mean Training Accuracy: ", landscapes_train_mean, "Mean Testing Accuracy: ", landscapes_test_mean, "Std Dev Training Accuracy: ", landscapes_train_std, "Std Dev Testing Accuracy: ", landscapes_test_std)

    landscapes_train_accuracy_svm = np.array(landscapes_train_accuracy_svm)
    landscapes_train_accuracy_svm = np.reshape(landscapes_train_accuracy_svm, newshape=(-1,1))
    landscapes_train_mean = np.mean(landscapes_train_accuracy_svm)
    landscapes_train_std = np.std(landscapes_train_accuracy_svm)

    landscapes_test_accuracy_svm = np.array(landscapes_test_accuracy_svm)
    landscapes_test_accuracy_svm = np.reshape(landscapes_test_accuracy_svm, newshape=(-1,1))
    landscapes_test_mean = np.mean(landscapes_test_accuracy_svm)
    landscapes_test_std = np.std(landscapes_test_accuracy_svm)

    print("landscapes Features - svm Model")
    print("Mean Training Accuracy: ", landscapes_train_mean, "Mean Testing Accuracy: ", landscapes_test_mean, "Std Dev Training Accuracy: ", landscapes_train_std, "Std Dev Testing Accuracy: ", landscapes_test_std)
    
    carlson_train_accuracy_ridge = np.array(carlson_train_accuracy_ridge)
    carlson_train_accuracy_ridge = np.reshape(carlson_train_accuracy_ridge, newshape=(-1,1))
    carlson_train_mean = np.mean(carlson_train_accuracy_ridge)
    carlson_train_std = np.std(carlson_train_accuracy_ridge)

    carlson_test_accuracy_ridge = np.array(carlson_test_accuracy_ridge)
    carlson_test_accuracy_ridge = np.reshape(carlson_test_accuracy_ridge, newshape=(-1,1))
    carlson_test_mean = np.mean(carlson_test_accuracy_ridge)
    carlson_test_std = np.std(carlson_test_accuracy_ridge)

    print("carlson Features - ridge Model")
    print("Mean Training Accuracy: ", carlson_train_mean, "Mean Testing Accuracy: ", carlson_test_mean, "Std Dev Training Accuracy: ", carlson_train_std, "Std Dev Testing Accuracy: ", carlson_test_std)

    carlson_train_accuracy_svm = np.array(carlson_train_accuracy_svm)
    carlson_train_accuracy_svm = np.reshape(carlson_train_accuracy_svm, newshape=(-1,1))
    carlson_train_mean = np.mean(carlson_train_accuracy_svm)
    carlson_train_std = np.std(carlson_train_accuracy_svm)

    carlson_test_accuracy_svm = np.array(carlson_test_accuracy_svm)
    carlson_test_accuracy_svm = np.reshape(carlson_test_accuracy_svm, newshape=(-1,1))
    carlson_test_mean = np.mean(carlson_test_accuracy_svm)
    carlson_test_std = np.std(carlson_test_accuracy_svm)

    print("carlson Features - svm Model")
    print("Mean Training Accuracy: ", carlson_train_mean, "Mean Testing Accuracy: ", carlson_test_mean, "Std Dev Training Accuracy: ", carlson_train_std, "Std Dev Testing Accuracy: ", carlson_test_std)

    kernel_train_accuracy_svm = np.array(kernel_train_accuracy_svm)
    kernel_train_accuracy_svm = np.reshape(kernel_train_accuracy_svm, newshape=(-1,1))
    kernel_train_mean = np.mean(kernel_train_accuracy_svm)
    kernel_train_std = np.std(kernel_train_accuracy_svm)

    kernel_test_accuracy_svm = np.array(kernel_test_accuracy_svm)
    kernel_test_accuracy_svm = np.reshape(kernel_test_accuracy_svm, newshape=(-1,1))
    kernel_test_mean = np.mean(kernel_test_accuracy_svm)
    kernel_test_std = np.std(kernel_test_accuracy_svm)

    print("kernel Features - svm Model")
    print("Mean Training Accuracy: ", kernel_train_mean, "Mean Testing Accuracy: ", kernel_test_mean, "Std Dev Training Accuracy: ", kernel_train_std, "Std Dev Testing Accuracy: ", kernel_test_std)

    results = pd.DataFrame()
    results['index'] = np.arange(start = 1,stop = 12)
    results['tent_train_accuracy_ridge'] = tent_train_accuracy_ridge
    results['tent_test_accuracy_ridge'] = tent_test_accuracy_ridge
    results['tent_train_accuracy_svm'] = tent_train_accuracy_svm
    results['tent_test_accuracy_svm'] = tent_test_accuracy_svm

    results['gmm_train_accuracy_ridge'] = gmm_train_accuracy_ridge
    results['gmm_test_accuracy_ridge'] = gmm_test_accuracy_ridge
    results['gmm_train_accuracy_svm'] = gmm_train_accuracy_svm
    results['gmm_test_accuracy_svm'] = gmm_test_accuracy_svm

    results['images_train_accuracy_ridge'] = images_train_accuracy_ridge
    results['images_test_accuracy_ridge'] = images_test_accuracy_ridge
    results['images_train_accuracy_svm'] = images_train_accuracy_svm
    results['images_test_accuracy_svm'] = images_test_accuracy_svm

    results['landscapes_train_accuracy_ridge'] = landscapes_train_accuracy_ridge
    results['landscapes_test_accuracy_ridge'] = landscapes_test_accuracy_ridge
    results['landscapes_train_accuracy_svm'] = landscapes_train_accuracy_svm
    results['landscapes_test_accuracy_svm'] = landscapes_test_accuracy_svm

    results['carlson_train_accuracy_ridge'] = carlson_train_accuracy_ridge
    results['carlson_test_accuracy_ridge'] = carlson_test_accuracy_ridge
    results['carlson_train_accuracy_svm'] = carlson_train_accuracy_svm
    results['carlson_test_accuracy_svm'] = carlson_test_accuracy_svm

    results['kernel_train_accuracy_svm'] = kernel_train_accuracy_svm
    results['kernel_test_accuracy_svm'] = kernel_test_accuracy_svm

    results.to_csv('protein_modeling_results_second_job.csv',index=False)
