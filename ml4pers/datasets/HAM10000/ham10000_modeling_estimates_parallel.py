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


def subsample(dgm, frac = .1, t = 4):
    dgm_diff = np.diff(dgm)
    keep = np.where(dgm_diff>t)[0]
    sample = np.where(dgm_diff<=t)[0]
    s_end = sample.size
    s_size = int(np.floor(s_end*frac))
    s_sample = np.random.choice(s_end, s_size,replace=False)
    s_index = np.hstack([keep, s_sample])
    dgm_sampled = dgm[s_index]
    return dgm_sampled

warnings.simplefilter("ignore")
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

Data0 = pickle.load(open("persistence/diagrams_0_1000.pickle", "rb"))
Data1 = pickle.load(open("persistence/diagrams_1000_2000.pickle", "rb"))
Data2 = pickle.load(open("persistence/diagrams_2000_3000.pickle", "rb"))
Data3 = pickle.load(open("persistence/diagrams_3000_4000.pickle", "rb"))
Data4 = pickle.load(open("persistence/diagrams_4000_5000.pickle", "rb"))
Data5 = pickle.load(open("persistence/diagrams_5000_6000.pickle", "rb"))
Data6 = pickle.load(open("persistence/diagrams_6000_7000.pickle", "rb"))
Data7 = pickle.load(open("persistence/diagrams_7000_8000.pickle", "rb"))
Data8 = pickle.load(open("persistence/diagrams_8000_9000.pickle", "rb"))
Data9 = pickle.load(open("persistence/diagrams_9000_10000.pickle", "rb"))

Data_full = pd.concat([Data0, Data1, Data2, Data3, Data4, Data5, Data6, Data7, Data8, Data9], axis=0, ignore_index=True)

Data_train, Data = train_test_split(Data_full, test_size = .0275, random_state=12, stratify = Data_full['labels'])

R0 = np.array(Data_train['dgm_R_rgb_0'])
G0 = np.array(Data_train['dgm_G_rgb_0'])
B0 = np.array(Data_train['dgm_B_rgb_0'])

H0 = np.array(Data_train['dgm_H_hsv_0'])
S0 = np.array(Data_train['dgm_S_hsv_0'])
V0 = np.array(Data_train['dgm_V_hsv_0'])

X0 = np.array(Data_train['dgm_X_xyz_0'])
Y0 = np.array(Data_train['dgm_Y_xyz_0'])
Z0 = np.array(Data_train['dgm_Z_xyz_0'])

R1 = np.array(Data_train['dgm_R_rgb_1'])
G1 = np.array(Data_train['dgm_G_rgb_1'])
B1 = np.array(Data_train['dgm_B_rgb_1'])

H1 = np.array(Data_train['dgm_H_hsv_1'])
S1 = np.array(Data_train['dgm_S_hsv_1'])
V1 = np.array(Data_train['dgm_V_hsv_1'])

X1 = np.array(Data_train['dgm_X_xyz_1'])
Y1 = np.array(Data_train['dgm_Y_xyz_1'])
Z1 = np.array(Data_train['dgm_Z_xyz_1'])

labels = np.array(Data_train['labels'])

R0_sample = []
for i in range(0, len(R0)):
    dgm = subsample(R0[i])
    R0_sample.append(dgm)
    
R1_sample = []
for i in range(0, len(R1)):
    dgm = subsample(R1[i])
    R1_sample.append(dgm)
    
G0_sample = []
for i in range(0, len(G0)):
    dgm = subsample(G0[i])
    G0_sample.append(dgm)
    
G1_sample = []
for i in range(0, len(G1)):
    dgm = subsample(G1[i])
    G1_sample.append(dgm)
    
B0_sample = []
for i in range(0, len(B0)):
    dgm = subsample(B0[i])
    B0_sample.append(dgm)
    
B1_sample = []
for i in range(0, len(B1)):
    dgm = subsample(B1[i])
    B1_sample.append(dgm)
    
H0_sample = []
for i in range(0, len(H0)):
    dgm = subsample(H0[i])
    H0_sample.append(dgm)
    
H1_sample = []
for i in range(0, len(H1)):
    dgm = subsample(H1[i])
    H1_sample.append(dgm)
    
S0_sample = []
for i in range(0, len(S0)):
    dgm = subsample(S0[i])
    S0_sample.append(dgm)
    
S1_sample = []
for i in range(0, len(S1)):
    dgm = subsample(S1[i])
    S1_sample.append(dgm)
    
V0_sample = []
for i in range(0, len(V0)):
    dgm = subsample(V0[i])
    V0_sample.append(dgm)
    
V1_sample = []
for i in range(0, len(V1)):
    dgm = subsample(V1[i])
    V1_sample.append(dgm)
    
X0_sample = []
for i in range(0, len(X0)):
    dgm = subsample(X0[i])
    X0_sample.append(dgm)
    
X1_sample = []
for i in range(0, len(X1)):
    dgm = subsample(X1[i])
    X1_sample.append(dgm)
    
Y0_sample = []
for i in range(0, len(Y0)):
    dgm = subsample(Y0[i])
    Y0_sample.append(dgm)
    
Y1_sample = []
for i in range(0, len(Y1)):
    dgm = subsample(Y1[i])
    Y1_sample.append(dgm)
    
Z0_sample = []
for i in range(0, len(Z0)):
    dgm = subsample(Z0[i])
    Z0_sample.append(dgm)
    
Z1_sample = []
for i in range(0, len(Z1)):
    dgm = subsample(Z1[i])
    Z1_sample.append(dgm)

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

# Determine Index
if rank == 0:
    states = np.arange(0,nruns*size)
    states = np.array_split(states, size)
else:
    states = []

states = comm.scatter(states, root = 0)

n = len(labels) 
obs = np.arange(0,n)
start = MPI.Wtime()
k = 0
for t in range(0,nruns):
    print("Starting Sample ", t)
    R0_train_sample, R0_test_sample, R1_train_sample, R1_test_sample, G0_train_sample, G0_test_sample, G1_train_sample, G1_test_sample, B0_train_sample, B0_test_sample, B1_train_sample, B1_test_sample, H0_train_sample, H0_test_sample, H1_train_sample, H1_test_sample, S0_train_sample, S0_test_sample, S1_train_sample, S1_test_sample, V0_train_sample, V0_test_sample, V1_train_sample, V1_test_sample, X0_train_sample, X0_test_sample, X1_train_sample, X1_test_sample, Y0_train_sample, Y0_test_sample, Y1_train_sample, Y1_test_sample, Z0_train_sample, Z0_test_sample, Z1_train_sample, Z1_test_sample,y_train, y_test = train_test_split(R0_sample, R1_sample, G0_sample, G1_sample, B0_sample, B1_sample, H0_sample, H1_sample, S0_sample, S1_sample, V0_sample, V1_sample, X0_sample, X1_sample, Y0_sample, Y1_sample, Z0_sample, Z1_sample, labels, test_size = .2, random_state=states[t], stratify = labels)

    ### Tent Features
    d = 10
    p = 1.5
    X_train_features_R0_tent, X_test_features_R0_tent = tent_features(R0_train_sample, R0_test_sample, d = d, padding = p)
    X_train_features_G0_tent, X_test_features_G0_tent = tent_features(G0_train_sample, G0_test_sample, d = d, padding = p)
    X_train_features_B0_tent, X_test_features_B0_tent = tent_features(B0_train_sample, B0_test_sample, d = d, padding = p)
    X_train_features_X0_tent, X_test_features_X0_tent = tent_features(X0_train_sample, X0_test_sample, d = d, padding = p)
    X_train_features_Y0_tent, X_test_features_Y0_tent = tent_features(Y0_train_sample, Y0_test_sample, d = d, padding = p)
    X_train_features_Z0_tent, X_test_features_Z0_tent = tent_features(Z0_train_sample, Z0_test_sample, d = d, padding = p)
    X_train_features_H0_tent, X_test_features_H0_tent = tent_features(H0_train_sample, H0_test_sample, d = d, padding = p)
    X_train_features_S0_tent, X_test_features_S0_tent = tent_features(S0_train_sample, S0_test_sample, d = d, padding = p)
    X_train_features_V0_tent, X_test_features_V0_tent = tent_features(V0_train_sample, V0_test_sample, d = d, padding = p)

    X_train_features_R1_tent, X_test_features_R1_tent = tent_features(R1_train_sample, R1_test_sample, d = d, padding = p)
    X_train_features_G1_tent, X_test_features_G1_tent = tent_features(G1_train_sample, G1_test_sample, d = d, padding = p)
    X_train_features_B1_tent, X_test_features_B1_tent = tent_features(B1_train_sample, B1_test_sample, d = d, padding = p)
    X_train_features_X1_tent, X_test_features_X1_tent = tent_features(X1_train_sample, X1_test_sample, d = d, padding = p)
    X_train_features_Y1_tent, X_test_features_Y1_tent = tent_features(Y1_train_sample, Y1_test_sample, d = d, padding = p)
    X_train_features_Z1_tent, X_test_features_Z1_tent = tent_features(Z1_train_sample, Z1_test_sample, d = d, padding = p)
    X_train_features_H1_tent, X_test_features_H1_tent = tent_features(H1_train_sample, H1_test_sample, d = d, padding = p)
    X_train_features_S1_tent, X_test_features_S1_tent = tent_features(S1_train_sample, S1_test_sample, d = d, padding = p)
    X_train_features_V1_tent, X_test_features_V1_tent = tent_features(V1_train_sample, V1_test_sample, d = d, padding = p)

    X_train_features = np.column_stack((X_train_features_R0_tent, X_train_features_G0_tent, X_train_features_B0_tent,X_train_features_X0_tent, X_train_features_Y0_tent, X_train_features_Z0_tent,X_train_features_H0_tent, X_train_features_S0_tent, X_train_features_V0_tent,X_train_features_R1_tent, X_train_features_G1_tent, X_train_features_B1_tent,X_train_features_X1_tent, X_train_features_Y1_tent, X_train_features_Z1_tent,X_train_features_H1_tent, X_train_features_S1_tent, X_train_features_V1_tent))
    X_test_features = np.column_stack((X_test_features_R0_tent, X_test_features_G0_tent, X_test_features_B0_tent,X_test_features_X0_tent, X_test_features_Y0_tent, X_test_features_Z0_tent,X_test_features_H0_tent, X_test_features_S0_tent, X_test_features_V0_tent,X_test_features_R1_tent, X_test_features_G1_tent, X_test_features_B1_tent,X_test_features_X1_tent, X_test_features_Y1_tent, X_test_features_Z1_tent,X_test_features_H1_tent, X_test_features_S1_tent, X_test_features_V1_tent))

    ### Ridge Model
    ridge_model = RidgeClassifier().fit(X_train_features, y_train)
    tent_train_accuracy_ridge[k] = ridge_model.score(X_train_features, y_train)
    tent_test_accuracy_ridge[k] = ridge_model.score(X_test_features, y_test)

    ### SVM Model
    c = 5
    svm_model = SVC(kernel='rbf', C = c).fit(X_train_features, y_train)
    tent_train_accuracy_svm[k] = svm_model.score(X_train_features, y_train)
    tent_test_accuracy_svm[k] = svm_model.score(X_test_features, y_test)

    ### CDER Adaptive
    X_train_features_R0_cder, X_test_features_R0_cder = adaptive_features(R0_train_sample, R0_test_sample, "cder", y_train)
    X_train_features_G0_cder, X_test_features_G0_cder = adaptive_features(G0_train_sample, G0_test_sample, "cder", y_train)
    X_train_features_B0_cder, X_test_features_B0_cder = adaptive_features(B0_train_sample, B0_test_sample, "cder", y_train)
    X_train_features_X0_cder, X_test_features_X0_cder = adaptive_features(X0_train_sample, X0_test_sample, "cder", y_train)
    X_train_features_Y0_cder, X_test_features_Y0_cder = adaptive_features(Y0_train_sample, Y0_test_sample, "cder", y_train)
    X_train_features_Z0_cder, X_test_features_Z0_cder = adaptive_features(Z0_train_sample, Z0_test_sample, "cder", y_train)
    X_train_features_H0_cder, X_test_features_H0_cder = adaptive_features(H0_train_sample, H0_test_sample, "cder", y_train)
    X_train_features_S0_cder, X_test_features_S0_cder = adaptive_features(S0_train_sample, S0_test_sample, "cder", y_train)
    X_train_features_V0_cder, X_test_features_V0_cder = adaptive_features(V0_train_sample, V0_test_sample, "cder", y_train)

    X_train_features_R1_cder, X_test_features_R1_cder = adaptive_features(R1_train_sample, R1_test_sample, "cder", y_train)
    X_train_features_G1_cder, X_test_features_G1_cder = adaptive_features(G1_train_sample, G1_test_sample, "cder", y_train)
    X_train_features_B1_cder, X_test_features_B1_cder = adaptive_features(B1_train_sample, B1_test_sample, "cder", y_train)
    X_train_features_X1_cder, X_test_features_X1_cder = adaptive_features(X1_train_sample, X1_test_sample, "cder", y_train)
    X_train_features_Y1_cder, X_test_features_Y1_cder = adaptive_features(Y1_train_sample, Y1_test_sample, "cder", y_train)
    X_train_features_Z1_cder, X_test_features_Z1_cder = adaptive_features(Z1_train_sample, Z1_test_sample, "cder", y_train)
    X_train_features_H1_cder, X_test_features_H1_cder = adaptive_features(H1_train_sample, H1_test_sample, "cder", y_train)
    X_train_features_S1_cder, X_test_features_S1_cder = adaptive_features(S1_train_sample, S1_test_sample, "cder", y_train)
    X_train_features_V1_cder, X_test_features_V1_cder = adaptive_features(V1_train_sample, V1_test_sample, "cder", y_train)

    X_train_features = np.column_stack((X_train_features_R1_cder, X_train_features_G1_cder, X_train_features_B1_cder,X_train_features_X1_cder, X_train_features_Y1_cder, X_train_features_Z1_cder,X_train_features_H1_cder, X_train_features_S1_cder, X_train_features_V1_cder, X_train_features_R0_cder, X_train_features_G0_cder, X_train_features_B0_cder,X_train_features_X0_cder, X_train_features_Y0_cder, X_train_features_Z0_cder,X_train_features_H0_cder, X_train_features_S0_cder, X_train_features_V0_cder))
    X_test_features = np.column_stack((X_test_features_R1_cder, X_test_features_G1_cder, X_test_features_B1_cder,X_test_features_X1_cder, X_test_features_Y1_cder, X_test_features_Z1_cder,X_test_features_H1_cder, X_test_features_S1_cder, X_test_features_V1_cder, X_test_features_R0_cder, X_test_features_G0_cder, X_test_features_B0_cder,X_test_features_X0_cder, X_test_features_Y0_cder, X_test_features_Z0_cder,X_test_features_H0_cder, X_test_features_S0_cder, X_test_features_V0_cder))
    ridge_model = RidgeClassifier().fit(X_train_features, y_train)

    gmm_train_accuracy_ridge[k] = ridge_model.score(X_train_features, y_train)
    gmm_test_accuracy_ridge[k] = ridge_model.score(X_test_features, y_test)

    svm_model = SVC(kernel='rbf', C = 1).fit(X_train_features, y_train)

    gmm_train_accuracy_svm[k] = svm_model.score(X_train_features, y_train)
    gmm_test_accuracy_svm[k] = svm_model.score(X_test_features, y_test)

    ### Persistence Images

    pixels = [[15,15],[20,20]]
    spread = [.5,1]
    i = 1
    j = 1
    X_train_features_R0_imgs, X_test_features_R0_imgs = persistence_image_features(R0_train_sample, R0_test_sample, pixels = pixels[i], spread = spread[j])
    X_train_features_G0_imgs, X_test_features_G0_imgs = persistence_image_features(G0_train_sample, G0_test_sample, pixels = pixels[i], spread = spread[j])
    X_train_features_B0_imgs, X_test_features_B0_imgs = persistence_image_features(B0_train_sample, B0_test_sample, pixels = pixels[i], spread = spread[j])
    X_train_features_X0_imgs, X_test_features_X0_imgs = persistence_image_features(X0_train_sample, X0_test_sample, pixels = pixels[i], spread = spread[j])
    X_train_features_Y0_imgs, X_test_features_Y0_imgs = persistence_image_features(Y0_train_sample, Y0_test_sample, pixels = pixels[i], spread = spread[j])
    X_train_features_Z0_imgs, X_test_features_Z0_imgs = persistence_image_features(Z0_train_sample, Z0_test_sample, pixels = pixels[i], spread = spread[j])
    X_train_features_H0_imgs, X_test_features_H0_imgs = persistence_image_features(H0_train_sample, H0_test_sample, pixels = pixels[i], spread = spread[j])
    X_train_features_S0_imgs, X_test_features_S0_imgs = persistence_image_features(S0_train_sample, S0_test_sample, pixels = pixels[i], spread = spread[j])
    X_train_features_V0_imgs, X_test_features_V0_imgs = persistence_image_features(V0_train_sample, V0_test_sample, pixels = pixels[i], spread = spread[j])

    X_train_features_R1_imgs, X_test_features_R1_imgs = persistence_image_features(R1_train_sample, R1_test_sample, pixels = pixels[i], spread = spread[j])
    X_train_features_G1_imgs, X_test_features_G1_imgs = persistence_image_features(G1_train_sample, G1_test_sample, pixels = pixels[i], spread = spread[j])
    X_train_features_B1_imgs, X_test_features_B1_imgs = persistence_image_features(B1_train_sample, B1_test_sample, pixels = pixels[i], spread = spread[j])
    X_train_features_X1_imgs, X_test_features_X1_imgs = persistence_image_features(X1_train_sample, X1_test_sample, pixels = pixels[i], spread = spread[j])
    X_train_features_Y1_imgs, X_test_features_Y1_imgs = persistence_image_features(Y1_train_sample, Y1_test_sample, pixels = pixels[i], spread = spread[j])
    X_train_features_Z1_imgs, X_test_features_Z1_imgs = persistence_image_features(Z1_train_sample, Z1_test_sample, pixels = pixels[i], spread = spread[j])
    X_train_features_H1_imgs, X_test_features_H1_imgs = persistence_image_features(H1_train_sample, H1_test_sample, pixels = pixels[i], spread = spread[j])
    X_train_features_S1_imgs, X_test_features_S1_imgs = persistence_image_features(S1_train_sample, S1_test_sample, pixels = pixels[i], spread = spread[j])
    X_train_features_V1_imgs, X_test_features_V1_imgs = persistence_image_features(V1_train_sample, V1_test_sample, pixels = pixels[i], spread = spread[j])

    X_train_features = np.column_stack((X_train_features_R0_imgs, X_train_features_G0_imgs, X_train_features_B0_imgs,X_train_features_X0_imgs, X_train_features_Y0_imgs, X_train_features_Z0_imgs,X_train_features_H0_imgs, X_train_features_S0_imgs, X_train_features_V0_imgs,X_train_features_R1_imgs, X_train_features_G1_imgs, X_train_features_B1_imgs,X_train_features_X1_imgs, X_train_features_Y1_imgs, X_train_features_Z1_imgs,X_train_features_H1_imgs, X_train_features_S1_imgs, X_train_features_V1_imgs))
    X_test_features = np.column_stack((X_test_features_R0_imgs, X_test_features_G0_imgs, X_test_features_B0_imgs,X_test_features_X0_imgs, X_test_features_Y0_imgs, X_test_features_Z0_imgs,X_test_features_H0_imgs, X_test_features_S0_imgs, X_test_features_V0_imgs,X_test_features_R1_imgs, X_test_features_G1_imgs, X_test_features_B1_imgs,X_test_features_X1_imgs, X_test_features_Y1_imgs, X_test_features_Z1_imgs,X_test_features_H1_imgs, X_test_features_S1_imgs, X_test_features_V1_imgs))
    ridge_model = RidgeClassifier().fit(X_train_features, y_train)
    images_train_accuracy_ridge[k] = ridge_model.score(X_train_features, y_train)
    images_test_accuracy_ridge[k] = ridge_model.score(X_test_features, y_test)

    svm_model = SVC(kernel='rbf', C=1).fit(X_train_features, y_train)
    images_train_accuracy_svm[k] = svm_model.score(X_train_features, y_train)
    images_test_accuracy_svm[k] = svm_model.score(X_test_features, y_test)

    ### Landscapes
    i = 10
    j = 50
    X_train_features_R0_landscapes, X_test_features_R0_landscapes = landscape_features(R0_train_sample, R0_test_sample, num_landscapes = i, resolution = j)
    X_train_features_G0_landscapes, X_test_features_G0_landscapes = landscape_features(G0_train_sample, G0_test_sample, num_landscapes = i, resolution = j)
    X_train_features_B0_landscapes, X_test_features_B0_landscapes = landscape_features(B0_train_sample, B0_test_sample, num_landscapes = i, resolution = j)
    X_train_features_X0_landscapes, X_test_features_X0_landscapes = landscape_features(X0_train_sample, X0_test_sample, num_landscapes = i, resolution = j)
    X_train_features_Y0_landscapes, X_test_features_Y0_landscapes = landscape_features(Y0_train_sample, Y0_test_sample, num_landscapes = i, resolution = j)
    X_train_features_Z0_landscapes, X_test_features_Z0_landscapes = landscape_features(Z0_train_sample, Z0_test_sample, num_landscapes = i, resolution = j)
    X_train_features_H0_landscapes, X_test_features_H0_landscapes = landscape_features(H0_train_sample, H0_test_sample, num_landscapes = i, resolution = j)
    X_train_features_S0_landscapes, X_test_features_S0_landscapes = landscape_features(S0_train_sample, S0_test_sample, num_landscapes = i, resolution = j)
    X_train_features_V0_landscapes, X_test_features_V0_landscapes = landscape_features(V0_train_sample, V0_test_sample, num_landscapes = i, resolution = j)
    
    X_train_features_R1_landscapes, X_test_features_R1_landscapes = landscape_features(R1_train_sample, R1_test_sample, num_landscapes = i, resolution = j)
    X_train_features_G1_landscapes, X_test_features_G1_landscapes = landscape_features(G1_train_sample, G1_test_sample, num_landscapes = i, resolution = j)
    X_train_features_B1_landscapes, X_test_features_B1_landscapes = landscape_features(B1_train_sample, B1_test_sample, num_landscapes = i, resolution = j)
    X_train_features_X1_landscapes, X_test_features_X1_landscapes = landscape_features(X1_train_sample, X1_test_sample, num_landscapes = i, resolution = j)
    X_train_features_Y1_landscapes, X_test_features_Y1_landscapes = landscape_features(Y1_train_sample, Y1_test_sample, num_landscapes = i, resolution = j)
    X_train_features_Z1_landscapes, X_test_features_Z1_landscapes = landscape_features(Z1_train_sample, Z1_test_sample, num_landscapes = i, resolution = j)
    X_train_features_H1_landscapes, X_test_features_H1_landscapes = landscape_features(H1_train_sample, H1_test_sample, num_landscapes = i, resolution = j)
    X_train_features_S1_landscapes, X_test_features_S1_landscapes = landscape_features(S1_train_sample, S1_test_sample, num_landscapes = i, resolution = j)
    X_train_features_V1_landscapes, X_test_features_V1_landscapes = landscape_features(V1_train_sample, V1_test_sample, num_landscapes = i, resolution = j)
    X_train_features = np.column_stack((X_train_features_R1_landscapes, X_train_features_R0_landscapes,X_train_features_G1_landscapes,X_train_features_G0_landscapes, X_train_features_B1_landscapes,X_train_features_B0_landscapes,X_train_features_X1_landscapes,X_train_features_X0_landscapes, X_train_features_Y1_landscapes, X_train_features_Y0_landscapes,X_train_features_Z1_landscapes,X_train_features_Z0_landscapes,X_train_features_H1_landscapes,X_train_features_H0_landscapes, X_train_features_S1_landscapes,X_train_features_S0_landscapes, X_train_features_V1_landscapes, X_train_features_V0_landscapes))
    X_test_features = np.column_stack((X_test_features_R1_landscapes, X_test_features_R0_landscapes, X_test_features_G1_landscapes, X_test_features_G0_landscapes,X_test_features_B1_landscapes,X_test_features_B0_landscapes,X_test_features_X1_landscapes, X_test_features_X0_landscapes,X_test_features_Y1_landscapes, X_test_features_Y0_landscapes,X_test_features_Z1_landscapes,X_test_features_Z0_landscapes,X_test_features_H1_landscapes, X_test_features_H0_landscapes,X_test_features_S1_landscapes, X_test_features_S0_landscapes,X_test_features_V1_landscapes,X_test_features_V0_landscapes))
    ridge_model = RidgeClassifier().fit(X_train_features, y_train)
    landscapes_train_accuracy_ridge[k] = ridge_model.score(X_train_features, y_train)
    landscapes_test_accuracy_ridge[k] = ridge_model.score(X_test_features, y_test)

    svm_model = SVC(kernel='rbf', C=1).fit(X_train_features, y_train)
    landscapes_train_accuracy_svm[k] = svm_model.score(X_train_features, y_train)
    landscapes_test_accuracy_svm[k] = svm_model.score(X_test_features, y_test)

    ### Carlsson Coordinates

    R0_train_features1_cc1, R0_train_features1_cc2, R0_train_features1_cc3, R0_train_features1_cc4, R0_test_features1_cc1, R0_test_features1_cc2, R0_test_features1_cc3, R0_test_features1_cc4 = carlsson_coordinates(R0_train_sample, R0_test_sample)
    G0_train_features1_cc1, G0_train_features1_cc2, G0_train_features1_cc3, G0_train_features1_cc4, G0_test_features1_cc1, G0_test_features1_cc2, G0_test_features1_cc3, G0_test_features1_cc4 = carlsson_coordinates(G0_train_sample, G0_test_sample)
    B0_train_features1_cc1, B0_train_features1_cc2, B0_train_features1_cc3, B0_train_features1_cc4, B0_test_features1_cc1, B0_test_features1_cc2, B0_test_features1_cc3, B0_test_features1_cc4 = carlsson_coordinates(B0_train_sample, B0_test_sample)
    X0_train_features1_cc1, X0_train_features1_cc2, X0_train_features1_cc3, X0_train_features1_cc4, X0_test_features1_cc1, X0_test_features1_cc2, X0_test_features1_cc3, X0_test_features1_cc4 = carlsson_coordinates(X0_train_sample, X0_test_sample)
    Y0_train_features1_cc1, Y0_train_features1_cc2, Y0_train_features1_cc3, Y0_train_features1_cc4, Y0_test_features1_cc1, Y0_test_features1_cc2, Y0_test_features1_cc3, Y0_test_features1_cc4 = carlsson_coordinates(Y0_train_sample, Y0_test_sample)
    Z0_train_features1_cc1, Z0_train_features1_cc2, Z0_train_features1_cc3, Z0_train_features1_cc4, Z0_test_features1_cc1, Z0_test_features1_cc2, Z0_test_features1_cc3, Z0_test_features1_cc4 = carlsson_coordinates(Z0_train_sample, Z0_test_sample)
    H0_train_features1_cc1, H0_train_features1_cc2, H0_train_features1_cc3, H0_train_features1_cc4, H0_test_features1_cc1, H0_test_features1_cc2, H0_test_features1_cc3, H0_test_features1_cc4 = carlsson_coordinates(H0_train_sample, H0_test_sample)
    S0_train_features1_cc1, S0_train_features1_cc2, S0_train_features1_cc3, S0_train_features1_cc4, S0_test_features1_cc1, S0_test_features1_cc2, S0_test_features1_cc3, S0_test_features1_cc4 = carlsson_coordinates(S0_train_sample, S0_test_sample)
    V0_train_features1_cc1, V0_train_features1_cc2, V0_train_features1_cc3, V0_train_features1_cc4, V0_test_features1_cc1, V0_test_features1_cc2, V0_test_features1_cc3, V0_test_features1_cc4 = carlsson_coordinates(V0_train_sample, V0_test_sample)

    R1_train_features1_cc1, R1_train_features1_cc2, R1_train_features1_cc3, R1_train_features1_cc4, R1_test_features1_cc1, R1_test_features1_cc2, R1_test_features1_cc3, R1_test_features1_cc4 = carlsson_coordinates(R1_train_sample, R1_test_sample)
    G1_train_features1_cc1, G1_train_features1_cc2, G1_train_features1_cc3, G1_train_features1_cc4, G1_test_features1_cc1, G1_test_features1_cc2, G1_test_features1_cc3, G1_test_features1_cc4 = carlsson_coordinates(G1_train_sample, G1_test_sample)
    B1_train_features1_cc1, B1_train_features1_cc2, B1_train_features1_cc3, B1_train_features1_cc4, B1_test_features1_cc1, B1_test_features1_cc2, B1_test_features1_cc3, B1_test_features1_cc4 = carlsson_coordinates(B1_train_sample, B1_test_sample)
    X1_train_features1_cc1, X1_train_features1_cc2, X1_train_features1_cc3, X1_train_features1_cc4, X1_test_features1_cc1, X1_test_features1_cc2, X1_test_features1_cc3, X1_test_features1_cc4 = carlsson_coordinates(X1_train_sample, X1_test_sample)
    Y1_train_features1_cc1, Y1_train_features1_cc2, Y1_train_features1_cc3, Y1_train_features1_cc4, Y1_test_features1_cc1, Y1_test_features1_cc2, Y1_test_features1_cc3, Y1_test_features1_cc4 = carlsson_coordinates(Y1_train_sample, Y1_test_sample)
    Z1_train_features1_cc1, Z1_train_features1_cc2, Z1_train_features1_cc3, Z1_train_features1_cc4, Z1_test_features1_cc1, Z1_test_features1_cc2, Z1_test_features1_cc3, Z1_test_features1_cc4 = carlsson_coordinates(Z1_train_sample, Z1_test_sample)
    H1_train_features1_cc1, H1_train_features1_cc2, H1_train_features1_cc3, H1_train_features1_cc4, H1_test_features1_cc1, H1_test_features1_cc2, H1_test_features1_cc3, H1_test_features1_cc4 = carlsson_coordinates(H1_train_sample, H1_test_sample)
    S1_train_features1_cc1, S1_train_features1_cc2, S1_train_features1_cc3, S1_train_features1_cc4, S1_test_features1_cc1, S1_test_features1_cc2, S1_test_features1_cc3, S1_test_features1_cc4 = carlsson_coordinates(S1_train_sample, S1_test_sample)
    V1_train_features1_cc1, V1_train_features1_cc2, V1_train_features1_cc3, V1_train_features1_cc4, V1_test_features1_cc1, V1_test_features1_cc2, V1_test_features1_cc3, V1_test_features1_cc4 = carlsson_coordinates(V1_train_sample, V1_test_sample)


    X_train_features = np.column_stack((R1_train_features1_cc1, R1_train_features1_cc2, R1_train_features1_cc3, R1_train_features1_cc4, G1_train_features1_cc1, G1_train_features1_cc2, G1_train_features1_cc3, G1_train_features1_cc4, B1_train_features1_cc1, B1_train_features1_cc2, B1_train_features1_cc3, B1_train_features1_cc4,X1_train_features1_cc1, X1_train_features1_cc2, X1_train_features1_cc3, X1_train_features1_cc4,Y1_train_features1_cc1, Y1_train_features1_cc2, Y1_train_features1_cc3, Y1_train_features1_cc4, Z1_train_features1_cc1, Z1_train_features1_cc2, Z1_train_features1_cc3, Z1_train_features1_cc4, H1_train_features1_cc1, H1_train_features1_cc2, H1_train_features1_cc3, H1_train_features1_cc4, S1_train_features1_cc1, S1_train_features1_cc2, S1_train_features1_cc3, S1_train_features1_cc4, V1_train_features1_cc1, V1_train_features1_cc2, V1_train_features1_cc3, V1_train_features1_cc4))
    X_test_features = np.column_stack((R1_test_features1_cc1, R1_test_features1_cc2, R1_test_features1_cc3, R1_test_features1_cc4, G1_test_features1_cc1, G1_test_features1_cc2, G1_test_features1_cc3, G1_test_features1_cc4, B1_test_features1_cc1, B1_test_features1_cc2, B1_test_features1_cc3, B1_test_features1_cc4, X1_test_features1_cc1, X1_test_features1_cc2, X1_test_features1_cc3, X1_test_features1_cc4, Y1_test_features1_cc1, Y1_test_features1_cc2, Y1_test_features1_cc3, Y1_test_features1_cc4, Z1_test_features1_cc1, Z1_test_features1_cc2, Z1_test_features1_cc3, Z1_test_features1_cc4, H1_test_features1_cc1, H1_test_features1_cc2, H1_test_features1_cc3, H1_test_features1_cc4, S1_test_features1_cc1, S1_test_features1_cc2, S1_test_features1_cc3, S1_test_features1_cc4, V1_test_features1_cc1, V1_test_features1_cc2, V1_test_features1_cc3, V1_test_features1_cc4))
    ridge_model = RidgeClassifier(normalize=True).fit(X_train_features, y_train)
    carlson_train_accuracy_ridge[k] = ridge_model.score(X_train_features, y_train)
    carlson_test_accuracy_ridge[k] = ridge_model.score(X_test_features, y_test)

    svm_model = SVC(kernel='rbf', C=1).fit(X_train_features, y_train)
    carlson_train_accuracy_svm[k] = svm_model.score(X_train_features, y_train)
    carlson_test_accuracy_svm[k] = svm_model.score(X_test_features, y_test)

    k += 1

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

    results = pd.DataFrame()
    results['index'] = np.arange(start = 0,stop = nruns*size)
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

    results.to_csv('ham10000_results_no_kernel.csv',index=False)
