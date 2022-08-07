import numpy as np
import pandas as pd
import os
import json

np.set_printoptions(precision=2)

# -------------- IMPORT DATA --------------------------------------------------

# Importante notes:
# This data set was provided to me by Dr. Elizabeth Munch's lab. 
# This data set was originally provided by Dr. Ulrich Bauer and it corresponds 
# the data used in A Stable Multi-Scale Kernel for Topological Machine Learnin
# by Jan Reininghaus, Stefan Huber, Ulrich Bauer, Roland Kwit

# -----------------------------------------------------------------------------

# Decompres data set, this line is specific for linux!
os.system('tar -xzvf shrec14.tar.gz')
os.system('mv shrec14 data')
os.system('touch data/.gitignore')
os.system('echo "*" >> data/.gitignore')

Data = pd.read_csv('data/Uli_data.csv')

# Code to reshape the data in the groupby command below
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

# This part of the code is just testing the correct readin gof the data

for i in range(1,11):

	freq = i

	SampleDF = DgmsDF[DgmsDF.trial == freq].sample(frac=1)

	print(SampleDF.head(5))
