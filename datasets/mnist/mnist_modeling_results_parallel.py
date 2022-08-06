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

from keras.datasets import mnist
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
import dionysus as d
import math
from sklearn.preprocessing import scale

import matplotlib.pyplot as plt

from gudhi.representations.vector_methods import Landscape

from persim import PersImage
from persim import heat

sys.path.append('../persistence_methods')
import ATS
from persistence_methods_numba import reshape_persistence_diagrams
from persistence_methods import kernel_features
from persistence_methods import tent_features
from persistence_methods import carlsson_coordinates
from persistence_methods import adaptive_features
from persistence_methods import landscape_features
from persistence_methods import persistence_image_features
from mpi4py import MPI

warnings.simplefilter("ignore")
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

if rank == 0:
    (train_X, train_y), (test_X, test_y) = mnist.load_data()
else:
    0 == 0

def directional_transform(img):
    z = np.array([0,1])
    z = z.reshape([1,2])
    left_to_right = np.zeros((28,28))
    for i in range(0,28):
        for j in range(0,28):
            if img[i,j] != 0:
                left_to_right[i,j] = np.inner([i,j],z)
            else:
                left_to_right[i,j] = 0

    z = np.array([0,-1])
    z = z.reshape([1,2])
    right_to_left = np.zeros((28,28))
    for i in range(0,28):
        for j in range(0,28):
            if img[i,j] != 0:
                right_to_left[i,j] = abs(28 + np.inner([i,j],z))
            else:
                right_to_left[i,j] = 0

    z = np.array([1,0])
    z = z.reshape([1,2])
    bottom_to_top = np.zeros((28,28))
    for i in range(0,28):
        for j in range(0,28):
            if img[i,j] != 0:
                bottom_to_top[i,j] = np.inner([i,j],z)
            else:
                bottom_to_top[i,j] = 0

    z = np.array([-1,0])
    z = z.reshape([1,2])
    top_to_bottom = np.zeros((28,28))
    for i in range(0,28):
        for j in range(0,28):
            if img[i,j] != 0:
                top_to_bottom[i,j] = abs(28 + np.inner([i,j],z))
            else:
                top_to_bottom[i,j] = 0
    imgs = [left_to_right, right_to_left, bottom_to_top, top_to_bottom]
    return imgs

def compute_persistence(directional_transform):
    f_lower_star = d.fill_freudenthal(directional_transform, reverse = True)
    p = d.homology_persistence(f_lower_star)
    dgms_lower = d.init_diagrams(p, f_lower_star)
    return dgms_lower

def append_dim_list(dgms, dim_list):
    jth_pt = []
    for k in range(0, len(dgms)):
        if dgms[k].death - dgms[k].birth >=0:
            birth = dgms[k].birth
            death = dgms[k].death
        else:
            birth = dgms[k].death
            death = dgms[k].birth
        if math.isinf(death):
            b = 50
        else:
            b = death
        t = [birth, b]
        jth_pt.append(t)
    dim_list.append(np.array(jth_pt))

def fill_missing(dim_1):
    for i in range(0, len(dim_1)):
        if len(dim_1[i])== 0:
            dim_1[i] = np.array([[0,.01]])
        else: 
            dim_1[i] = dim_1[i]
    return dim_1

nruns = 1
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

carlson_train_accuracy_ridge = np.zeros(nruns)
carlson_test_accuracy_ridge = np.zeros(nruns)

carlson_train_accuracy_svm = np.zeros(nruns)
carlson_test_accuracy_svm = np.zeros(nruns)


if rank == 0:
    states = np.arange(0,nruns*size)
    states = np.array_split(states, size)
else:
    states = []

states = comm.scatter(states, root = 0)

if rank == 0:
    n = len(train_y)
    obs = np.arange(0,n)
   # obs, unused_obs = train_test_split(obs, random_state = 12, test_size = .8, stratify = train_y) 
    train_y = train_y[obs]
    zero_dim_0 = []
    zero_dim_1 = []
    zero_dim_2 = []
    zero_dim_3 = []
    one_dim_0 = []
    one_dim_1 = []
    one_dim_2 = []
    one_dim_3 = []
    for i in obs:
        print("Transforming Image ", i)
        img = train_X[i]
        imgs = directional_transform(img)
        for j in range(0,4):
            dgms_lower= compute_persistence(imgs[j])
            if j == 0:
                append_dim_list(dgms_lower[0], zero_dim_0)
                append_dim_list(dgms_lower[1], one_dim_0)
            if j == 1:
                append_dim_list(dgms_lower[0], zero_dim_1)
                append_dim_list(dgms_lower[1], one_dim_1)
            if j == 2:
                append_dim_list(dgms_lower[0], zero_dim_2)
                append_dim_list(dgms_lower[1], one_dim_2)
            if j == 3:
                append_dim_list(dgms_lower[0], zero_dim_3)
                append_dim_list(dgms_lower[1], one_dim_3)   

    one_dim_0 = fill_missing(one_dim_0)
    one_dim_1 = fill_missing(one_dim_1)
    one_dim_2 = fill_missing(one_dim_2)
    one_dim_3 = fill_missing(one_dim_3)

else:
    zero_dim_0 = []
    zero_dim_1 = []
    zero_dim_2 = []
    zero_dim_3 = []
    one_dim_0 = []
    one_dim_1 = []
    one_dim_2 = []
    one_dim_3 = []
    train_y = []

zero_dim_0 = comm.bcast(zero_dim_0, root = 0)
zero_dim_1 = comm.bcast(zero_dim_1, root = 0)
zero_dim_2 = comm.bcast(zero_dim_2, root = 0)
zero_dim_3 = comm.bcast(zero_dim_3, root = 0)
one_dim_0 = comm.bcast(one_dim_0, root = 0)
one_dim_1 = comm.bcast(one_dim_1, root = 0)
one_dim_2 = comm.bcast(one_dim_2, root = 0)
one_dim_3 = comm.bcast(one_dim_3, root = 0)
train_y = comm.bcast(train_y, root = 0)

n = len(train_y) 
obs = np.arange(0,n)
start = MPI.Wtime()
for t in range(0,nruns):
    print("Starting Sample ", t)
    train_index, test_index, y_train, y_test = train_test_split(obs, train_y, random_state = states[t], test_size = .2, stratify = train_y)

    zero_dim_ltr_train = np.array(zero_dim_0)[train_index]
    zero_dim_rtl_train = np.array(zero_dim_1)[train_index]
    zero_dim_ttb_train = np.array(zero_dim_2)[train_index]
    zero_dim_btt_train = np.array(zero_dim_3)[train_index]

    zero_dim_ltr_test = np.array(zero_dim_0)[test_index]
    zero_dim_rtl_test = np.array(zero_dim_1)[test_index]
    zero_dim_ttb_test = np.array(zero_dim_2)[test_index]
    zero_dim_btt_test = np.array(zero_dim_3)[test_index]

    one_dim_ltr_train = np.array(one_dim_0)[train_index]
    one_dim_rtl_train = np.array(one_dim_1)[train_index]
    one_dim_ttb_train = np.array(one_dim_2)[train_index]
    one_dim_btt_train = np.array(one_dim_3)[train_index]

    one_dim_ltr_test = np.array(one_dim_0)[test_index]
    one_dim_rtl_test = np.array(one_dim_1)[test_index]
    one_dim_ttb_test = np.array(one_dim_2)[test_index]
    one_dim_btt_test = np.array(one_dim_3)[test_index]

    ### Tent Features
    d = 10
    p = 1.2
    X_train_features_0_ltr_tent, X_test_features_0_ltr_tent = tent_features(zero_dim_ltr_train, zero_dim_ltr_test,d = d, padding = p)
    X_train_features_0_rtl_tent, X_test_features_0_rtl_tent = tent_features(zero_dim_rtl_train, zero_dim_rtl_test,d = d, padding = p)
    X_train_features_0_btt_tent, X_test_features_0_btt_tent = tent_features(zero_dim_btt_train, zero_dim_btt_test,d = d, padding = p)
    X_train_features_0_ttb_tent, X_test_features_0_ttb_tent = tent_features(zero_dim_ttb_train, zero_dim_ttb_test,d = d, padding = p)
    X_train_features_1_ltr_tent, X_test_features_1_ltr_tent = tent_features(one_dim_ltr_train, one_dim_ltr_test,d = d, padding = p)
    X_train_features_1_rtl_tent, X_test_features_1_rtl_tent = tent_features(one_dim_rtl_train, one_dim_rtl_test,d = d, padding = p)
    X_train_features_1_btt_tent, X_test_features_1_btt_tent = tent_features(one_dim_btt_train, one_dim_btt_test,d = d, padding = p)
    X_train_features_1_ttb_tent, X_test_features_1_ttb_tent = tent_features(one_dim_ttb_train, one_dim_ttb_test,d = d, padding = p)
    X_train_features = np.column_stack((X_train_features_1_ltr_tent,X_train_features_1_rtl_tent,X_train_features_1_ttb_tent,X_train_features_1_btt_tent,X_train_features_0_ltr_tent,X_train_features_0_rtl_tent,X_train_features_0_btt_tent,X_train_features_0_ttb_tent))
    X_test_features = np.column_stack((X_test_features_1_ltr_tent,X_test_features_1_rtl_tent,X_test_features_1_ttb_tent,X_test_features_1_btt_tent,X_test_features_0_ltr_tent,X_test_features_0_rtl_tent,X_test_features_0_btt_tent,X_test_features_0_ttb_tent))

    ### Ridge Model
    ridge_model = RidgeClassifier().fit(X_train_features, y_train)
    tent_train_accuracy_ridge[t] = ridge_model.score(X_train_features, y_train)
    tent_test_accuracy_ridge[t] = ridge_model.score(X_test_features, y_test)

    ### SVM Model
    c = 5
    svm_model = SVC(kernel='rbf', C = c).fit(X_train_features, y_train)
    tent_train_accuracy_svm[t] = svm_model.score(X_train_features, y_train)
    tent_test_accuracy_svm[t] = svm_model.score(X_test_features, y_test)

    ### Adaptive Features
    d = 35
    X_train_features_0_ltr_gmm, X_test_features_0_ltr_gmm = adaptive_features(zero_dim_ltr_train, zero_dim_ltr_test, model="gmm", y_train=y_train, d = d)
    X_train_features_0_rtl_gmm, X_test_features_0_rtl_gmm = adaptive_features(zero_dim_rtl_train, zero_dim_rtl_test, model="gmm", y_train=y_train, d = d)
    X_train_features_0_ttb_gmm, X_test_features_0_ttb_gmm = adaptive_features(zero_dim_ttb_train, zero_dim_ttb_test, model="gmm", y_train=y_train, d = d)
    X_train_features_0_btt_gmm, X_test_features_0_btt_gmm = adaptive_features(zero_dim_btt_train, zero_dim_btt_test, model="gmm", y_train=y_train, d = d)

    X_train_features_1_ltr_gmm, X_test_features_1_ltr_gmm = adaptive_features(one_dim_ltr_train, one_dim_ltr_test, model="gmm", y_train=y_train, d = d)
    X_train_features_1_rtl_gmm, X_test_features_1_rtl_gmm = adaptive_features(one_dim_rtl_train, one_dim_rtl_test, model="gmm", y_train=y_train, d = d)
    X_train_features_1_ttb_gmm, X_test_features_1_ttb_gmm = adaptive_features(one_dim_ttb_train, one_dim_ttb_test, model="gmm", y_train=y_train, d = d)
    X_train_features_1_btt_gmm, X_test_features_1_btt_gmm = adaptive_features(one_dim_btt_train, one_dim_btt_test, model="gmm", y_train=y_train, d = d)
    X_train_features = np.column_stack((X_train_features_1_ltr_gmm,X_train_features_1_rtl_gmm,X_train_features_1_ttb_gmm,X_train_features_1_btt_gmm,X_train_features_0_ltr_gmm,X_train_features_0_rtl_gmm,X_train_features_0_btt_gmm,X_train_features_0_ttb_gmm))
    X_test_features = np.column_stack((X_test_features_1_ltr_gmm,X_test_features_1_rtl_gmm,X_test_features_1_ttb_gmm,X_test_features_1_btt_gmm,X_test_features_0_ltr_gmm,X_test_features_0_rtl_gmm,X_test_features_0_btt_gmm,X_test_features_0_ttb_gmm))

    ridge_model = RidgeClassifier().fit(X_train_features, y_train)
    gmm_train_accuracy_ridge[t] = ridge_model.score(X_train_features, y_train)
    gmm_test_accuracy_ridge[t] = ridge_model.score(X_test_features, y_test)

    c = 10
    svm_model = SVC(kernel='rbf', C=c).fit(X_train_features, y_train)
    gmm_train_accuracy_svm[t] = svm_model.score(X_train_features, y_train)
    gmm_test_accuracy_svm[t] = svm_model.score(X_test_features, y_test)

    ### Persistence images
    p = [15,15]
    s = 1
    X_train_features_0_ltr_imgs, X_test_features_0_ltr_imgs = persistence_image_features(zero_dim_ltr_train, zero_dim_ltr_test, pixels=p, spread=s)
    X_train_features_0_rtl_imgs, X_test_features_0_rtl_imgs = persistence_image_features(zero_dim_rtl_train, zero_dim_rtl_test, pixels=p, spread=s)
    X_train_features_0_ttb_imgs, X_test_features_0_ttb_imgs = persistence_image_features(zero_dim_ttb_train, zero_dim_ttb_test, pixels=p, spread=s)
    X_train_features_0_btt_imgs, X_test_features_0_btt_imgs = persistence_image_features(zero_dim_btt_train, zero_dim_btt_test, pixels=p, spread=s)

    X_train_features_1_ltr_imgs, X_test_features_1_ltr_imgs = persistence_image_features(one_dim_ltr_train, one_dim_ltr_test, pixels=p, spread=s)
    X_train_features_1_rtl_imgs, X_test_features_1_rtl_imgs = persistence_image_features(one_dim_rtl_train, one_dim_rtl_test, pixels=p, spread=s)
    X_train_features_1_ttb_imgs, X_test_features_1_ttb_imgs = persistence_image_features(one_dim_ttb_train, one_dim_ttb_test, pixels=p, spread=s)
    X_train_features_1_btt_imgs, X_test_features_1_btt_imgs = persistence_image_features(one_dim_btt_train, one_dim_btt_test, pixels=p, spread=s)
    
    X_train_features = np.column_stack((X_train_features_1_ltr_imgs,X_train_features_1_rtl_imgs,X_train_features_1_ttb_imgs,X_train_features_1_btt_imgs,X_train_features_0_ltr_imgs,X_train_features_0_rtl_imgs,X_train_features_0_btt_imgs,X_train_features_0_ttb_imgs))
    X_test_features = np.column_stack((X_test_features_1_ltr_imgs,X_test_features_1_rtl_imgs,X_test_features_1_ttb_imgs,X_test_features_1_btt_imgs,X_test_features_0_ltr_imgs,X_test_features_0_rtl_imgs,X_test_features_0_btt_imgs,X_test_features_0_ttb_imgs))

    ridge_model = RidgeClassifier().fit(X_train_features, y_train)
    images_train_accuracy_ridge[t] = ridge_model.score(X_train_features, y_train)
    images_test_accuracy_ridge[t] = ridge_model.score(X_test_features, y_test)

    c = 10
    svm_model = SVC(kernel='rbf', C = c).fit(X_train_features, y_train)
    images_train_accuracy_svm[t] = svm_model.score(X_train_features, y_train)
    images_test_accuracy_svm[t] = svm_model.score(X_test_features, y_test)

    ### Landscape Features
    i = 3
    j = 50
    X_train_features_1_ltr_landscapes, X_test_features_1_ltr_landscapes = landscape_features(one_dim_ltr_train, one_dim_ltr_test, num_landscapes=i, resolution=j)
    X_train_features_0_ltr_landscapes, X_test_features_0_ltr_landscapes = landscape_features(zero_dim_ltr_train, zero_dim_ltr_test, num_landscapes=i, resolution=j)

    X_train_features_1_rtl_landscapes, X_test_features_1_rtl_landscapes = landscape_features(one_dim_rtl_train, one_dim_rtl_test, num_landscapes=i, resolution=j)
    X_train_features_0_rtl_landscapes, X_test_features_0_rtl_landscapes = landscape_features(zero_dim_rtl_train, zero_dim_rtl_test, num_landscapes=i, resolution=j)

    X_train_features_1_ttb_landscapes, X_test_features_1_ttb_landscapes = landscape_features(one_dim_ttb_train, one_dim_ttb_test, num_landscapes=i, resolution=j)
    X_train_features_0_ttb_landscapes, X_test_features_0_ttb_landscapes = landscape_features(zero_dim_ttb_train, zero_dim_ttb_test, num_landscapes=i, resolution=j)

    X_train_features_1_btt_landscapes, X_test_features_1_btt_landscapes = landscape_features(one_dim_btt_train, one_dim_btt_test, num_landscapes=i, resolution=j)
    X_train_features_0_btt_landscapes, X_test_features_0_btt_landscapes = landscape_features(zero_dim_btt_train, zero_dim_btt_test, num_landscapes=i, resolution=j)
    
    X_train_features = np.column_stack((X_train_features_1_ltr_landscapes,X_train_features_1_rtl_landscapes,X_train_features_1_ttb_landscapes,X_train_features_1_btt_landscapes,X_train_features_0_ltr_landscapes,X_train_features_0_rtl_landscapes,X_train_features_0_btt_landscapes,X_train_features_0_ttb_landscapes))
    X_test_features = np.column_stack((X_test_features_1_ltr_landscapes,X_test_features_1_rtl_landscapes,X_test_features_1_ttb_landscapes,X_test_features_1_btt_landscapes,X_test_features_0_ltr_landscapes,X_test_features_0_rtl_landscapes,X_test_features_0_btt_landscapes,X_test_features_0_ttb_landscapes))

    ridge_model = RidgeClassifier().fit(X_train_features, y_train)
    landscapes_train_accuracy_ridge[t] = ridge_model.score(X_train_features, y_train)
    landscapes_test_accuracy_ridge[t] = ridge_model.score(X_test_features, y_test)

    c = 20
    svm_model = SVC(kernel='rbf', C=c).fit(X_train_features, y_train)
    landscapes_train_accuracy_svm[t] = svm_model.score(X_train_features, y_train)
    landscapes_test_accuracy_svm[t] = svm_model.score(X_test_features, y_test)

    ### Carlson Coordinates
    c = 50

    X_train_features_0_ltr_cc1, X_train_features_0_ltr_cc2, X_train_features_0_ltr_cc3, X_train_features_0_ltr_cc4, X_test_features_0_ltr_cc1, X_test_features_0_ltr_cc2, X_test_features_0_ltr_cc3, X_test_features_0_ltr_cc4 = carlsson_coordinates(zero_dim_ltr_train, zero_dim_ltr_test)
    X_train_features_1_ltr_cc1, X_train_features_1_ltr_cc2, X_train_features_1_ltr_cc3, X_train_features_1_ltr_cc4, X_test_features_1_ltr_cc1, X_test_features_1_ltr_cc2, X_test_features_1_ltr_cc3, X_test_features_1_ltr_cc4 = carlsson_coordinates(one_dim_ltr_train, one_dim_ltr_test)
    X_train_features_0_rtl_cc1, X_train_features_0_rtl_cc2, X_train_features_0_rtl_cc3, X_train_features_0_rtl_cc4, X_test_features_0_rtl_cc1, X_test_features_0_rtl_cc2, X_test_features_0_rtl_cc3, X_test_features_0_rtl_cc4 = carlsson_coordinates(zero_dim_rtl_train, zero_dim_rtl_test)
    X_train_features_1_rtl_cc1, X_train_features_1_rtl_cc2, X_train_features_1_rtl_cc3, X_train_features_1_rtl_cc4, X_test_features_1_rtl_cc1, X_test_features_1_rtl_cc2, X_test_features_1_rtl_cc3, X_test_features_1_rtl_cc4 = carlsson_coordinates(one_dim_rtl_train, one_dim_rtl_test)
    X_train_features_0_btt_cc1, X_train_features_0_btt_cc2, X_train_features_0_btt_cc3, X_train_features_0_btt_cc4, X_test_features_0_btt_cc1, X_test_features_0_btt_cc2, X_test_features_0_btt_cc3, X_test_features_0_btt_cc4 = carlsson_coordinates(zero_dim_btt_train, zero_dim_btt_test)
    X_train_features_1_btt_cc1, X_train_features_1_btt_cc2, X_train_features_1_btt_cc3, X_train_features_1_btt_cc4, X_test_features_1_btt_cc1, X_test_features_1_btt_cc2, X_test_features_1_btt_cc3, X_test_features_1_btt_cc4 = carlsson_coordinates(one_dim_btt_train, one_dim_btt_test)
    X_train_features_0_ttb_cc1, X_train_features_0_ttb_cc2, X_train_features_0_ttb_cc3, X_train_features_0_ttb_cc4, X_test_features_0_ttb_cc1, X_test_features_0_ttb_cc2, X_test_features_0_ttb_cc3, X_test_features_0_ttb_cc4 = carlsson_coordinates(zero_dim_ttb_train, zero_dim_ttb_test)
    X_train_features_1_ttb_cc1, X_train_features_1_ttb_cc2, X_train_features_1_ttb_cc3, X_train_features_1_ttb_cc4, X_test_features_1_ttb_cc1, X_test_features_1_ttb_cc2, X_test_features_1_ttb_cc3, X_test_features_1_ttb_cc4 = carlsson_coordinates(one_dim_ttb_train, one_dim_ttb_test)

    X_train_features = np.column_stack((scale(X_train_features_0_ltr_cc1), scale(X_train_features_0_ltr_cc2),scale(X_train_features_0_ltr_cc3),scale(X_train_features_0_ltr_cc4),
                                   scale(X_train_features_0_rtl_cc1), scale(X_train_features_0_rtl_cc2),scale(X_train_features_0_rtl_cc3),scale(X_train_features_0_rtl_cc4),
                                   scale(X_train_features_0_ttb_cc1), scale(X_train_features_0_ttb_cc2),scale(X_train_features_0_ttb_cc3),scale(X_train_features_0_ttb_cc4),
                                   scale(X_train_features_0_btt_cc1), scale(X_train_features_0_btt_cc2),scale(X_train_features_0_btt_cc3),scale(X_train_features_0_btt_cc4),
                                   scale(X_train_features_1_ltr_cc1), scale(X_train_features_1_ltr_cc2),scale(X_train_features_1_ltr_cc3),scale(X_train_features_1_ltr_cc4),
                                   scale(X_train_features_1_rtl_cc1), scale(X_train_features_1_rtl_cc2),scale(X_train_features_1_rtl_cc3),scale(X_train_features_1_rtl_cc4),
                                   scale(X_train_features_1_ttb_cc1), scale(X_train_features_1_ttb_cc2),scale(X_train_features_1_ttb_cc3),scale(X_train_features_1_ttb_cc4),
                                   scale(X_train_features_1_btt_cc1), scale(X_train_features_1_btt_cc2),scale(X_train_features_1_btt_cc3),scale(X_train_features_1_btt_cc4)))

    X_test_features = np.column_stack((scale(X_test_features_0_ltr_cc1), scale(X_test_features_0_ltr_cc2), scale(X_test_features_0_ltr_cc3), scale(X_test_features_0_ltr_cc4),
                                  scale(X_test_features_0_rtl_cc1), scale(X_test_features_0_rtl_cc2), scale(X_test_features_0_rtl_cc3), scale(X_test_features_0_rtl_cc4),
                                   scale(X_test_features_0_ttb_cc1), scale(X_test_features_0_ttb_cc2), scale(X_test_features_0_ttb_cc3), scale(X_test_features_0_ttb_cc4),
                                  scale(X_test_features_0_btt_cc1), scale(X_test_features_0_btt_cc2), scale(X_test_features_0_btt_cc3), scale(X_test_features_0_btt_cc4),
                                  scale(X_test_features_1_ltr_cc1), scale(X_test_features_1_ltr_cc2), scale(X_test_features_1_ltr_cc3), scale(X_test_features_1_ltr_cc4),
                                  scale(X_test_features_1_rtl_cc1), scale(X_test_features_1_rtl_cc2), scale(X_test_features_1_rtl_cc3), scale(X_test_features_1_rtl_cc4),
                                   scale(X_test_features_1_ttb_cc1), scale(X_test_features_1_ttb_cc2), scale(X_test_features_1_ttb_cc3), scale(X_test_features_1_ttb_cc4),
                                  scale(X_test_features_1_btt_cc1), scale(X_test_features_1_btt_cc2), scale(X_test_features_1_btt_cc3), scale(X_test_features_1_btt_cc4)))

    ridge_model = RidgeClassifier().fit(X_train_features, y_train)
    carlson_train_accuracy_ridge[t] = ridge_model.score(X_train_features, y_train)
    carlson_test_accuracy_ridge[t] = ridge_model.score(X_test_features, y_test)

    svm_model = SVC(kernel='rbf', gamma = .125).fit(X_train_features, y_train)
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
