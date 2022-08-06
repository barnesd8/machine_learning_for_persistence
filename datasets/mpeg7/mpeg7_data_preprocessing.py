import numpy as np
import pandas as pd
import os
import os.path
import matplotlib.image as mpimg
import cv2
import pickle
from scipy.spatial import distance
import math

from scipy import sparse
from cv2 import findContours
from scipy.ndimage import binary_erosion, binary_dilation
from scipy.ndimage.morphology import binary_fill_holes
from ripser import ripser

import json

np.set_printoptions(precision=2)

# -------------- IMPORT DATA --------------------------------------------------

# Decompres data set, this line is specific for linux!
### Need to add to check if there is a .gitignore already
if os.path.exists("data/"):
    print("Data already unzipped")
else:
    os.system('tar -xzvf mpeg7.tar.gz')
    os.system('mv mpeg7 data')
    os.system('touch data/.gitignore')
    os.system('echo "*" >> data/.gitignore')

# Load after file is unzipped
with open("../config.json") as json_config_file:
    filepaths= json.load(json_config_file)
fp = filepaths['fp']
data = fp + filepaths['mpeg7data']
pers = fp + filepaths['mpeg7pers']
contours = fp + filepaths['mpeg7contours']
os.system('mkdir ' + contours)
os.system('touch contours/.gitignore')
os.system('echo "*" >> contours/.gitignore')


# -----------------------------------------------------------------------------
# Importante notes:
# The implementatio nhere used to compute contours was made by Sarah Tymochko.

def get_contour(img):

    # Very hacky image processing to get rid of outlying pixels and fill holes in region
    img = binary_erosion(img)
    img = binary_dilation(img)
    img = binary_fill_holes(img)*1
    
    # Necessary to get data in right format for findContours
    img = np.ascontiguousarray(img, dtype=np.uint8)

    # Get the outline of the shapes
    # _, contours, _ = findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    
    # Reformat contour
    pt_cloud = np.array([list(i[0]) for i in contours[0]])
    
    return pt_cloud

def get_dist_curve(pt_cloud):
    center = [np.mean(pt_cloud[:,0]), np.mean(pt_cloud[:,1])]
    
    return np.sqrt([np.sum((pt_cloud[i]-center)**2) for i in range(len(pt_cloud))])

def scale_time(df):
    max_t = max([len(dlist) for dlist in df['DistCurve']])

    for i in df.index:
        df.loc[i,'t'] = list(np.linspace(0,len(df.loc[i,'DistCurve'])/max_t,len(df.loc[i,'DistCurve'])))
        
    return df

l = os.listdir(data)
l.remove('.gitignore')

point_clouds = []
distance_curves = []

subsampling_countours = 500


for file in l:
#for i in range(198,210):
 #   file = l[i]
    filename = os.fsdecode(file)

    img = mpimg.imread(data+filename)

    if len(np.shape(img)) == 3:    
        img = img[:,:,0]

    pt_cloud = get_contour(img)
    pt_cloud_filename = contours + filename[:-3] + "csv"
    np.savetxt(pt_cloud_filename, pt_cloud, delimiter = ",")

    if pt_cloud.shape[0] > subsampling_countours:
        ind = np.random.choice(np.arange(pt_cloud.shape[0]), subsampling_countours)
        ind.sort()

        pt_cloud = pt_cloud[ind,:] 

    point_clouds.append(pt_cloud)

    distance_curves.append(get_dist_curve(pt_cloud))

# Compute persistent diagrams 
diagrams = pd.DataFrame(index=range(len(point_clouds)), columns=['Name', 'Outline', 'DistCurve', 'persistence_outline', 'lower_star_persistence'])


for j in range(len(point_clouds)):
    print(j)
    d = ripser(point_clouds[j], maxdim=1, do_cocycles=False)['dgms']

    file = l[j]
    filename = os.fsdecode(file)
    name = filename[:-4]

    # SUBLEVEL SET PERSISTENCE

    N = len(distance_curves[j])

    I = np.arange(N-1)
    J = np.arange(1, N)
    V = np.maximum(distance_curves[j][0:-1], distance_curves[j][1::])
    # Add vertex birth times along the diagonal of the distance matrix
    I = np.concatenate((I, np.arange(N)))
    J = np.concatenate((J, np.arange(N)))
    V = np.concatenate((V, distance_curves[j]))
    #Create the sparse distance matrix
    D = sparse.coo_matrix((V, (I, J)), shape=(N, N)).tocsr()
    dgm0 = ripser(D, maxdim=0, distance_matrix=True)['dgms'][0]

    diagrams.loc[j] = [name, point_clouds[j], distance_curves[j], d, dgm0]


# Save with pickle
os.system('mkdir ' + pers)
os.system('touch persistence/.gitignore')
os.system('echo "*" >> persistence/.gitignore')
with open(pers + 'diagrams.pickle', 'wb') as handle:
    pickle.dump(diagrams, handle, protocol=pickle.HIGHEST_PROTOCOL)