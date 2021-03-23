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

Data = pd.read_csv('data/Uli_data.csv')

def reshapeVec(g):
    A = np.array([g.dim,g.birth,g.death])
    A = A.T
    return A

DgmsDF = Data.groupby(['freq', 'trial']).apply(reshapeVec)
DgmsDF = DgmsDF.reset_index()
DgmsDF = DgmsDF.rename(columns = {0:'CollectedDgm'})

def getDgm(A, dim = 0):
    if type(dim) != str:
        if dim == 0:
            A = A[np.where(np.logical_or(A[:,0] == dim, A[:,0] == -1))[0],1:]
            
        if dim == 1:
            A = A[np.where(np.logical_or(A[:,0] == dim, A[:,0] == -2))[0],1:]
    
    return(A)

DgmsDF['Dgm1'] = DgmsDF.CollectedDgm.apply(lambda x: getDgm(x, dim = 1))
DgmsDF['Dgm0'] = DgmsDF.CollectedDgm.apply(lambda x: getDgm(x, dim = 0))
DgmsDF['DgmInf'] = DgmsDF.CollectedDgm.apply(lambda x: getDgm(x, dim = 'essential'))

def label(index):
    if 0 <= index <= 19:
        return 'male_neutral'
    elif 20<= index <=39:
        return 'male_bodybuilder'
    elif 40<= index <=59:
        return 'male_fat'
    elif 60<= index <=79:
        return 'male_thin'
    elif 80<= index <=99:
        return 'male_average'
    elif 100<= index <=119:
        return 'female_neutral'
    elif 120<= index <=139:
        return 'female_bodybuilder'
    elif 140<= index <=159:
        return 'female_fat'
    elif 160<= index <=179:
        return 'female_thin'
    elif 180<= index <=199:
        return 'female_average'
    elif 200<= index <=219:
        return 'child_neutral'
    elif 220<= index <=239:
        return 'child_bodybuilder'
    elif 240<= index <=259:
        return 'child_fat'
    elif 260<= index <=279:
        return 'child_thin'
    elif 280<= index <=299:
        return 'child_average'
    else:
        print('What are you giving me?')

DgmsDF['TrainingLabel'] = DgmsDF.freq.apply(label)
DgmsDF= DgmsDF.sample(frac=1)

T1 = DgmsDF[DgmsDF.trial==7]
X_dgm0 = np.array(T1['Dgm0'])
X_dgm1 = np.array(T1['Dgm1'])

labels = np.array(T1['TrainingLabel'])
print(type(labels))
#labels = [d.split('_',1)[0] for d in labels]
labels = pd.DataFrame(labels)
label_names = labels.copy()

label_unique = pd.DataFrame(labels)
label_unique = label_unique[0].unique()

i=0
for l in label_unique:
    labels[labels == l]=i
    i += 1

labels = labels[0].tolist()
label_names = label_names[0].tolist()

nruns = 20
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

n = len(labels)
index = np.arange(0,n)
tuning_index, training_index = train_test_split(index, test_size = .7, random_state=12, stratify = labels)
X_dgm0= X_dgm0[training_index]
X_dgm1 = X_dgm1[training_index]
labels = np.array(labels)[training_index]

if rank == 0:
    states = np.arange(0,nruns*size)
    states = np.array_split(states, size)
else:
    states = []

states = comm.scatter(states, root = 0)

start = MPI.Wtime()
for t in range(0,nruns):
    print("Starting Sample ", t)
    X_dgm0_train, X_dgm0_test, X_dgm1_train, X_dgm1_test, y_train, y_test = train_test_split(X_dgm0, X_dgm1, labels, random_state = states[t], test_size = .2, stratify = labels)

    ### Tent Features
    d = 12
    p = 1.2
    X_train_features_1_tent, X_test_features_1_tent = tent_features(X_dgm1_train, X_dgm1_test, d, p)
    X_train_features_0_tent, X_test_features_0_tent = tent_features(X_dgm0_train, X_dgm0_test, d, p)
    X_train_features = np.column_stack((X_train_features_0_tent, X_train_features_1_tent))
    X_test_features = np.column_stack((X_test_features_0_tent, X_test_features_1_tent))

    ### Ridge Model
    ridge_model = RidgeClassifier().fit(X_train_features, y_train)
    tent_train_accuracy_ridge[t] = ridge_model.score(X_train_features, y_train)
    tent_test_accuracy_ridge[t] = ridge_model.score(X_test_features, y_test)

    ### SVM Model
    c = 1
    svm_model = SVC(kernel='rbf', C = c).fit(X_train_features, y_train)
    tent_train_accuracy_svm[t] = svm_model.score(X_train_features, y_train)
    tent_test_accuracy_svm[t] = svm_model.score(X_test_features, y_test)

    ### Adaptive Features
    X_train_features_1_gmm, X_test_features_1_gmm = adaptive_features(X_dgm1_train, X_dgm1_test, "cder", y_train)
    X_train_features_0_gmm, X_test_features_0_gmm = adaptive_features(X_dgm0_train, X_dgm0_test, "cder", y_train)
    X_train_features = np.column_stack((X_train_features_0_gmm, X_train_features_1_gmm))
    X_test_features = np.column_stack((X_test_features_0_gmm, X_test_features_1_gmm))

    ridge_model = RidgeClassifier().fit(X_train_features, y_train)
    gmm_train_accuracy_ridge[t] = ridge_model.score(X_train_features, y_train)
    gmm_test_accuracy_ridge[t] = ridge_model.score(X_test_features, y_test)

    c = 10
    svm_model = SVC(kernel='rbf', C=c).fit(X_train_features, y_train)
    gmm_train_accuracy_svm[t] = svm_model.score(X_train_features, y_train)
    gmm_test_accuracy_svm[t] = svm_model.score(X_test_features, y_test)

    ### Persistence images
    pixels = [40,40]
    spread = .5
    X_train_features_1_imgs, X_test_features_1_imgs = persistence_image_features(X_dgm1_train, X_dgm1_test,pixels=[50,50], spread = 1)
    X_train_features_0_imgs, X_test_features_0_imgs = persistence_image_features(X_dgm0_train, X_dgm0_test,pixels=[50,50], spread = 1)
    X_train_features = np.column_stack((X_train_features_1_imgs,X_train_features_0_imgs))
    X_test_features = np.column_stack((X_test_features_1_imgs,X_test_features_0_imgs))

    ridge_model = RidgeClassifier().fit(X_train_features, y_train)
    images_train_accuracy_ridge[t] = ridge_model.score(X_train_features, y_train)
    images_test_accuracy_ridge[t] = ridge_model.score(X_test_features, y_test)

    c = 1
    svm_model = SVC(kernel='rbf', C = c).fit(X_train_features, y_train)
    images_train_accuracy_svm[t] = svm_model.score(X_train_features, y_train)
    images_test_accuracy_svm[t] = svm_model.score(X_test_features, y_test)

    ### Kernel Features
    s = .5
    X_train_features_1_kernel, X_test_features_1_kernel = fast_kernel_features(X_dgm1_train, X_dgm1_test, .5)
    X_train_features_0_kernel, X_test_features_0_kernel = fast_kernel_features(X_dgm0_train, X_dgm0_test, .5)
    X_train_features = X_train_features_1_kernel + X_train_features_0_kernel
    X_test_features = X_test_features_1_kernel + X_test_features_0_kernel
    svm_model = NuSVC(kernel='precomputed').fit(X_train_features, y_train)
    kernel_train_accuracy_svm[t] = svm_model.score(X_train_features, y_train)
    kernel_test_accuracy_svm[t] = svm_model.score(X_test_features, y_test)

    ### Landscape Features
    n = 5
    r = 200
    X_train_features_1_landscapes, X_test_features_1_landscapes = landscape_features(X_dgm1_train, X_dgm1_test, num_landscapes=n, resolution=r)
    X_train_features_0_landscapes, X_test_features_0_landscapes = landscape_features(X_dgm0_train, X_dgm0_test, num_landscapes=n, resolution=r)
    X_train_features = np.column_stack((X_train_features_0_landscapes, X_train_features_1_landscapes))
    X_test_features = np.column_stack((X_test_features_0_landscapes, X_test_features_1_landscapes))

    ridge_model = RidgeClassifier().fit(X_train_features, y_train)
    landscapes_train_accuracy_ridge[t] = ridge_model.score(X_train_features, y_train)
    landscapes_test_accuracy_ridge[t] = ridge_model.score(X_test_features, y_test)

    c = 10
    svm_model = SVC(kernel='rbf', C=c).fit(X_train_features, y_train)
    landscapes_train_accuracy_svm[t] = svm_model.score(X_train_features, y_train)
    landscapes_test_accuracy_svm[t] = svm_model.score(X_test_features, y_test)

    ### Carlson Coordinates
    c = 50
    X_train_features1_cc1, X_train_features1_cc2, X_train_features1_cc3, X_train_features1_cc4, X_test_features1_cc1, X_test_features1_cc2, X_test_features1_cc3, X_test_features1_cc4 = carlsson_coordinates(X_dgm1_train, X_dgm1_test)
    X_train_features0_cc1, X_train_features0_cc2, X_train_features0_cc3, X_train_features0_cc4, X_test_features0_cc1, X_test_features0_cc2, X_test_features0_cc3, X_test_features0_cc4 = carlsson_coordinates(X_dgm0_train, X_dgm0_test)
    X_train_features = np.column_stack((X_train_features1_cc1, X_train_features1_cc2, X_train_features1_cc3, X_train_features1_cc4, X_train_features0_cc1, X_train_features0_cc2, X_train_features0_cc3, X_train_features0_cc4))
    X_test_features = np.column_stack((X_test_features1_cc1,X_test_features1_cc2, X_test_features1_cc3,X_test_features1_cc4, X_test_features0_cc1,X_test_features0_cc2, X_test_features0_cc3,X_test_features0_cc4))

    ridge_model = RidgeClassifier().fit(X_train_features, y_train)
    carlson_train_accuracy_ridge[t] = ridge_model.score(X_train_features, y_train)
    carlson_test_accuracy_ridge[t] = ridge_model.score(X_test_features, y_test)

    svm_model = SVC(kernel='rbf', C = c).fit(X_train_features, y_train)
    carlson_train_accuracy_svm[t] = svm_model.score(X_train_features, y_train)
    carlson_test_accuracy_svm[t] = svm_model.score(X_test_features, y_test)

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
    
    