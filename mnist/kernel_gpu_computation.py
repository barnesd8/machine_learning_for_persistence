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
from persistence_methods_cuda import create_index
from persistence_methods_cuda import cuda_kernel_features_train
from persistence_methods_cuda import cuda_kernel_features_test

warnings.simplefilter("ignore")


(train_X, train_y), (test_X, test_y) = mnist.load_data()

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


n = len(train_y)
obs = np.arange(0,n)
obs, unused_obs = train_test_split(obs, random_state = 12, test_size = .8, stratify = train_y) 
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


n = len(train_y) 
obs = np.arange(0,n)
state = int(sys.argv[1])

train_index, test_index, y_train, y_test = train_test_split(obs, train_y, random_state = state, test_size = .2, stratify = train_y)

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

### Kernel Features
print('Starting Kernel Features')
start = time.time()
dummy_train = np.arange(len(y_train), dtype=np.float64)
dummy_test = np.arange(len(y_test), dtype=np.float64)

s = 1.2
zero_dim_ltr_train_ = reshape_persistence_diagrams(zero_dim_ltr_train)
zero_dim_rtl_train_ = reshape_persistence_diagrams(zero_dim_rtl_train)
zero_dim_ttb_train_ = reshape_persistence_diagrams(zero_dim_ttb_train)
zero_dim_btt_train_ = reshape_persistence_diagrams(zero_dim_btt_train)

one_dim_ltr_train_ = reshape_persistence_diagrams(one_dim_ltr_train)
one_dim_rtl_train_ = reshape_persistence_diagrams(one_dim_rtl_train)
one_dim_ttb_train_ = reshape_persistence_diagrams(one_dim_ttb_train)
one_dim_btt_train_ = reshape_persistence_diagrams(one_dim_btt_train)

zero_dim_ltr_test_ = reshape_persistence_diagrams(zero_dim_ltr_test)
zero_dim_rtl_test_ = reshape_persistence_diagrams(zero_dim_rtl_test)
zero_dim_ttb_test_ = reshape_persistence_diagrams(zero_dim_ttb_test)
zero_dim_btt_test_ = reshape_persistence_diagrams(zero_dim_btt_test)

one_dim_ltr_test_ = reshape_persistence_diagrams(one_dim_ltr_test)
one_dim_rtl_test_ = reshape_persistence_diagrams(one_dim_rtl_test)
one_dim_ttb_test_ = reshape_persistence_diagrams(one_dim_ttb_test)
one_dim_btt_test_ = reshape_persistence_diagrams(one_dim_btt_test)

index_train_0_ltr = create_index(zero_dim_ltr_train_, dummy_train)
index_test_0_ltr = create_index(zero_dim_ltr_test_, dummy_test)
index_train_0_rtl = create_index(zero_dim_rtl_train_, dummy_train)
index_test_0_rtl = create_index(zero_dim_rtl_test_, dummy_test)
index_train_0_ttb = create_index(zero_dim_ttb_train_, dummy_train)
index_test_0_ttb = create_index(zero_dim_ttb_test_, dummy_test)
index_train_0_btt = create_index(zero_dim_btt_train_, dummy_train)
index_test_0_btt = create_index(zero_dim_btt_test_, dummy_test)


index_train_1_ltr = create_index(one_dim_ltr_train_, dummy_train)
index_test_1_ltr = create_index(one_dim_ltr_test_, dummy_test)
index_train_1_rtl = create_index(one_dim_rtl_train_, dummy_train)
index_test_1_rtl = create_index(one_dim_rtl_test_, dummy_test)
index_train_1_ttb = create_index(one_dim_ttb_train_, dummy_train)
index_test_1_ttb = create_index(one_dim_ttb_test_, dummy_test)
index_train_1_btt = create_index(one_dim_btt_train_, dummy_train)
index_test_1_btt = create_index(one_dim_btt_test_, dummy_test)

X_train_features_0_ltr_kernel = cuda_kernel_features_train(zero_dim_ltr_train_, index_train_0_ltr,s)
X_train_features_0_rtl_kernel = cuda_kernel_features_train(zero_dim_rtl_train_, index_train_0_rtl,s)
X_train_features_0_ttb_kernel = cuda_kernel_features_train(zero_dim_ttb_train_, index_train_0_ttb,s)
X_train_features_0_btt_kernel = cuda_kernel_features_train(zero_dim_btt_train_, index_train_0_btt,s)

X_train_features_1_ltr_kernel = cuda_kernel_features_train(one_dim_ltr_train_, index_train_1_ltr,s)
X_train_features_1_rtl_kernel = cuda_kernel_features_train(one_dim_rtl_train_, index_train_1_rtl,s)
X_train_features_1_ttb_kernel = cuda_kernel_features_train(one_dim_ttb_train_, index_train_1_ttb,s)
X_train_features_1_btt_kernel = cuda_kernel_features_train(one_dim_btt_train_, index_train_1_btt,s)

X_test_features_0_ltr_kernel = cuda_kernel_features_test(zero_dim_ltr_train_, zero_dim_ltr_test_, index_train_0_ltr,index_test_0_ltr,s)
X_test_features_0_rtl_kernel = cuda_kernel_features_test(zero_dim_rtl_train_, zero_dim_rtl_test_, index_train_0_rtl,index_test_0_rtl,s)
X_test_features_0_ttb_kernel = cuda_kernel_features_test(zero_dim_ttb_train_, zero_dim_ttb_test_, index_train_0_ttb,index_test_0_ttb, s)
X_test_features_0_btt_kernel = cuda_kernel_features_test(zero_dim_btt_train_, zero_dim_btt_test_, index_train_0_btt,index_test_0_btt, s)

X_test_features_1_ltr_kernel = cuda_kernel_features_test(one_dim_ltr_train_, one_dim_ltr_test_, index_train_1_ltr,index_test_1_ltr,s)
X_test_features_1_rtl_kernel = cuda_kernel_features_test(one_dim_rtl_train_, one_dim_rtl_test_, index_train_1_rtl,index_test_1_rtl,s)
X_test_features_1_ttb_kernel = cuda_kernel_features_test(one_dim_ttb_train_, one_dim_ttb_test_, index_train_1_ttb,index_test_1_ttb,s)
X_test_features_1_btt_kernel = cuda_kernel_features_test(one_dim_btt_train_, one_dim_btt_test_, index_train_1_btt,index_test_1_btt,s)

print("Cuda time: ", time.time()-start)

train_kernel = X_train_features_1_btt_kernel + X_train_features_1_ttb_kernel + X_train_features_1_rtl_kernel + X_train_features_1_ltr_kernel + X_train_features_0_btt_kernel + X_train_features_0_ttb_kernel + X_train_features_0_rtl_kernel + X_train_features_0_ltr_kernel
test_kernel = X_test_features_1_btt_kernel + X_test_features_1_ttb_kernel + X_test_features_1_rtl_kernel + X_test_features_1_ltr_kernel + X_test_features_0_btt_kernel + X_test_features_0_ttb_kernel + X_test_features_0_rtl_kernel + X_test_features_0_ltr_kernel

svc_model = NuSVC(kernel='precomputed')
svc_model.fit(train_kernel, y_train)

kernel_numba_train_accuracy_svm = svc_model.score(train_kernel, y_train)
print("score numba train:", svc_model.score(train_kernel, y_train))
kernel_numba_test_accuracy_svm = svc_model.score(test_kernel, y_test)
print("score numba test:",svc_model.score(test_kernel, y_test))

kernel_numba_train_accuracy_svm = np.array(kernel_numba_train_accuracy_svm)
kernel_numba_train_accuracy_svm = np.reshape(kernel_numba_train_accuracy_svm, newshape=(-1,1))
kernel_numba_train_mean = np.mean(kernel_numba_train_accuracy_svm)
kernel_numba_train_std = np.std(kernel_numba_train_accuracy_svm)

kernel_numba_test_accuracy_svm = np.array(kernel_numba_test_accuracy_svm)
kernel_numba_test_accuracy_svm = np.reshape(kernel_numba_test_accuracy_svm, newshape=(-1,1))
kernel_numba_test_mean = np.mean(kernel_numba_test_accuracy_svm)
kernel_numba_test_std = np.std(kernel_numba_test_accuracy_svm)

print("kernel_numba Features - svm Model")
print("Mean Training Accuracy: ", kernel_numba_train_mean, "Mean Testing Accuracy: ", kernel_numba_test_mean, "Std Dev Training Accuracy: ", kernel_numba_train_std, "Std Dev Testing Accuracy: ", kernel_numba_test_std)

    
