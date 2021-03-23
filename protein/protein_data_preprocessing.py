import numpy as np
import os
import pickle
import json

from ripser import ripser
from persim import plot_diagrams

# -------------- IMPORT DATA --------------------------------------------------

# Importante notes:
#
# This data set was provided to me by Dr. Kelin Xia's lab. It was orignally 
# used in A topological approach for protein classification by Zixuan Cang, 
# Lin Mu, Kedi Wu, Kristopher Opron, Kelin Xia, and Guo-Wei Wei.

# -----------------------------------------------------------------------------

# Decompres data set, this line is specific for linux!
### Need to add to check if there is a .gitignore already
os.system('tar -xzvf protein_data.tar.gz')
os.system('mv protein_data data')
os.system('touch data/.gitignore')
os.system('echo "*" >> data/.gitignore')

# Load after file is unzipped
with open("../config.json") as json_config_file:
    filepaths= json.load(json_config_file)
fp = filepaths['fp']
data = fp + filepaths['proteindata']
pers = fp + filepaths['proteinpers']

# Compute persisitent diagrams 
diagrams = []

j = 0
for file in os.listdir(data + 'SCOP40mini/'):
    filename = os.fsdecode(file)
        
    open_file = open(data + 'SCOP40mini/' + filename)
    lines = open_file.read().splitlines()
    
    coordinates = []
    
    # This loop read the coordinate data in R^3
    for i in range(12, len(lines)-3):
        line = lines[i]

        if line.split()[0]=='ATOM':
            x = float(line[32:38])
            y = float(line[38:46])
            z = float(line[46:54])
                        
            coordinates.append([x, y, z])
            
        
    coordinates = np.array(coordinates)
    
    # Persistent homology computation using ripser
    if len(coordinates)>500:
        d = ripser(coordinates, maxdim=1, do_cocycles=False, n_perm=500)['dgms']
    else:
    	d = ripser(coordinates, maxdim=1, do_cocycles=False)['dgms']
    
    name = filename[:-4]
    
    temp = {}

    temp['name'] = name
    temp['h0'] = d[0]
    temp['h1'] = d[1]

    diagrams.append(temp)

    print(j)
    j += 1
        
# Save with pickle
os.system('mkdir ' + pers)
os.system('touch persistence/.gitignore')
os.system('echo "*" >> persistence/.gitignore')
with open(pers + 'diagrams.pickle', 'wb') as handle:
    pickle.dump(diagrams, handle, protocol=pickle.HIGHEST_PROTOCOL)


# IMPORTANT NOTICE:
#
# This script was provided to me by Dr. Zixuan Cang. 
#
# It parses the data and creates the training and testing sets as used in A 
# topological approach for protein classification by Zixuan Cang, Lin Mu, 
# Kedi Wu, Kristopher Opron, Kelin Xia, and Guo-Wei Wei.


# 1: +train 2: -train 3: +test 4:-test
# http://pongor.itk.ppke.hu/benchmark/#/Benchmark_data_formats
import numpy as np
import os

os.system('mkdir data/Index')

mat = np.empty([1357,55], int)
infile = open('data/CAST.txt')
lines = infile.read().splitlines()
for i in range(len(lines)):
    line = lines[i]
    a = line[7:].split()
    for j in range(55):
        mat[i,j] = int(a[j])

for i in range(55):
    print(i+1)
    TrainIndex = []
    TestIndex = []
    TrainLabel = []
    TestLabel = []
    for j in range(1357):
        if mat[j,i] == 1 or mat[j,i] == 2:
            TrainIndex.append(j)
            if mat[j,i] == 1:
                TrainLabel.append(1)
            elif mat[j,i] == 2:
                TrainLabel.append(-1)
        if mat[j,i] == 3 or mat[j,i] == 4:
            TestIndex.append(j)
            if mat[j,i] == 3:
                TestLabel.append(1)
            elif mat[j,i] == 4:
                TestLabel.append(-1)

    TrainIndex = np.asarray(TrainIndex, int)
    TestIndex = np.asarray(TestIndex, int)
    TrainLabel = np.asarray(TrainLabel, int)
    TestLabel = np.asarray(TestLabel, int)

    print(len(TrainIndex), np.sum(TrainLabel), len(TestIndex), np.sum(TestLabel), len(TrainIndex)+len(TestIndex))

    outfile = open('data/Index/TrainIndex'+str(i+1)+'.npy','wb')
    np.save(outfile, TrainIndex)
    outfile.close()
    outfile = open('data/Index/TrainLabel'+str(i+1)+'.npy','wb')
    np.save(outfile, TrainLabel)
    outfile.close()
    outfile = open('data/Index/TestIndex'+str(i+1)+'.npy','wb')
    np.save(outfile, TestIndex)
    outfile.close()
    outfile = open('data/Index/TestLabel'+str(i+1)+'.npy','wb')
    np.save(outfile, TestLabel)
    outfile.close()
