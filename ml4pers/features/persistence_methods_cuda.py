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
from numba import guvectorize
import math

def reshape_persistence_diagrams(dgm):
    dgm_reshape = np.array([])
    n = len(dgm)
    print(n)
    for i in range(0,n):
        t = np.repeat(i, len(dgm[i]))
        t = t.reshape(len(dgm[i]),1)
        t1 = np.concatenate((t,dgm[i]),1)
        if i == 0:
            dgm_reshape = t1
        else:
            dgm_reshape = np.append(dgm_reshape, t1, 0)
    return dgm_reshape

def create_index(dgm, dummy):
    n = dummy.shape[0]
    index = np.zeros((n,2))
    arr = dgm[:,0]
    for i in range(0,n):
        t = np.where(arr==i)
        index[i,0] = min(t[0])
        index[i,1] = max(t[0])+1
    return index

@guvectorize(["void(float64[:,:],  float64[:,:], float64, float64[:,:])",],"(m,n),(p,r),()->(p,p)", target="cuda")
def cuda_kernel_features_train(train, index, s, result):
    n_train = index.shape[0]
    for i in range(n_train):
        for j in range(i):
            dgm0 = train[int(index[i,0]):int(index[i,1]),1:3]
            dgm1 = train[int(index[j,0]):int(index[j,1]),1:3]
            kSigma0 = 0
            kSigma1 = 0
            kSigma2 = 0
            sigma = s
            for k in range(dgm0.shape[0]):
                p = dgm0[k,0:2]
                for l in range(dgm0.shape[0]):
                    q = dgm0[l,0:2]
                    qc = dgm0[l, 1::-1]
                    pq = (p[0] - q[0])**2 + (p[1] - q[1])**2
                    pqc = (p[0] - qc[0])**2 + (p[1] - qc[1])**2
                    kSigma0 += math.exp(-( pq) / (8 * sigma)) - math.exp(-(pqc) / (8 * sigma))
            for k in range(dgm1.shape[0]):
                p = dgm1[k,0:2]
                for l in range(dgm1.shape[0]):
                    q = dgm1[l,0:2]
                    qc = dgm1[l, 1::-1]
                    pq = (p[0] - q[0])**2 + (p[1] - q[1])**2
                    pqc = (p[0] - qc[0])**2 + (p[1] - qc[1])**2
                    kSigma1 += math.exp(-( pq) / (8 * sigma)) - math.exp(-(pqc) / (8 * sigma))
            for k in range(dgm0.shape[0]):
                p = dgm0[k,0:2]
                for l in range(dgm1.shape[0]):
                    q = dgm1[l,0:2]
                    qc = dgm1[l, 1::-1]
                    pq = (p[0] - q[0])**2 + (p[1] - q[1])**2
                    pqc = (p[0] - qc[0])**2 + (p[1] - qc[1])**2
                    kSigma2 += math.exp(-( pq) / (8 * sigma)) - math.exp(-(pqc) / (8 * sigma))

            kSigma0 = kSigma0/(8 * np.pi * sigma)
            kSigma1 = kSigma1/(8 * np.pi * sigma)
            kSigma2 = kSigma2/(8 * np.pi * sigma)
            result[i,j] = math.sqrt(kSigma1 + kSigma0-2*kSigma2)
            result[j,i] = math.sqrt(kSigma1 + kSigma0-2*kSigma2)

@guvectorize(["void(float64[:,:], float64[:,:], float64[:,:], float64[:,:],float64 ,float64[:,:])",],"(m,n),(l,n),(p,r),(q,r), ()->(q,p)", target="cuda")
def cuda_kernel_features_test(train, test, index_train, index_test, s, result):
    n_train = index_train.shape[0]
    n_test = index_test.shape[0]
    for i in range(n_train):
        for j in range(n_test):
            dgm0 = train[int(index_train[i,0]):int(index_train[i,1]),1:3]
            dgm1 = test[int(index_test[j,0]):int(index_test[j,1]),1:3]
            kSigma0 = 0
            kSigma1 = 0
            kSigma2 = 0
            sigma = s
            for k in range(dgm0.shape[0]):
                p = dgm0[k,0:2]
                for l in range(dgm0.shape[0]):
                    q = dgm0[l,0:2]
                    qc = dgm0[l, 1::-1]
                    pq = (p[0] - q[0])**2 + (p[1] - q[1])**2
                    pqc = (p[0] - qc[0])**2 + (p[1] - qc[1])**2
                    kSigma0 += math.exp(-( pq) / (8 * sigma)) - math.exp(-(pqc) / (8 * sigma))
            for k in range(dgm1.shape[0]):
                p = dgm1[k,0:2]
                for l in range(dgm1.shape[0]):
                    q = dgm1[l,0:2]
                    qc = dgm1[l, 1::-1]
                    pq = (p[0] - q[0])**2 + (p[1] - q[1])**2
                    pqc = (p[0] - qc[0])**2 + (p[1] - qc[1])**2
                    kSigma1 += math.exp(-( pq) / (8 * sigma)) - math.exp(-(pqc) / (8 * sigma))
            for k in range(dgm0.shape[0]):
                p = dgm0[k,0:2]
                for l in range(dgm1.shape[0]):
                    q = dgm1[l,0:2]
                    qc = dgm1[l, 1::-1]
                    pq = (p[0] - q[0])**2 + (p[1] - q[1])**2
                    pqc = (p[0] - qc[0])**2 + (p[1] - qc[1])**2
                    kSigma2 += math.exp(-( pq) / (8 * sigma)) - math.exp(-(pqc) / (8 * sigma))

            kSigma0 = kSigma0/(8 * np.pi * sigma)
            kSigma1 = kSigma1/(8 * np.pi * sigma)
            kSigma2 = kSigma2/(8 * np.pi * sigma)
            result[j,i] = math.sqrt(kSigma1 + kSigma0-2*kSigma2)