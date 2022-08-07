import numpy as np
import pandas as pd
import os
import os.path
import matplotlib.image as mpimg
import cv2
import pickle
import PIL
import csv
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import skimage
import dionysus as d
import math
import sys

from scipy.spatial.distance import cdist
from skimage.morphology import convex_hull_image
from scipy import sparse
from ripser import ripser, lower_star_img

import json
import copy

from mpi4py import MPI

np.set_printoptions(precision=2)

# -------------- FUNCTIONS ----------------------------------------------------

def getImageFiltration(I, T, filtration, a=None):
    '''
    This function computes a filtration of binary images for a given image I. 
    Each step in the filtration is defined by the following following formula for a fixed T and a.

    .. math::
        I^*_t(i,j) = \begin{cases}
            0   & \mbox{if } I^*(i,j) > a(1 - t/T)
            255 & \mbox{if } I^*(i,j) \leq a(1 - t/T)
        \end{cases}

    where $T^*$ is the the image I after being converted to gray scale.

    :type I: numpy.array
    :param I: n x m image.

    :type T: float
    :param T: End point for the filtration.

    :type filtration: list or numpy.array
    :param filtration: Collection of values 0 <= t <= T that will be used to build the filtration of binary images I^*_t

    :type a: float
    :param a: Default = None. Scale for the filtration. If the value is None this implementation uses the mean pixel intensity as the scale for the filtration.

    :retrun: numpy.array - 3D array representing the filtration of binary images. The dimensions correspond to height x width x step_in_filtration.
    '''

    # Make sure input is an array

    image = np.array(I)

    # Make the original images gray scale
    if I.ndim > 2:
        img_grey = np.asarray(PIL.Image.fromarray(image).convert('L'))

        # my_img_grey = np.remainder((1/3)*(image[:,:,0] + image[:,:,1] + image[:,:,2]), 255)
    else:
        img_grey = image

    #the filtration calues can not go over T
    if np.max(filtration) > T:
        print('ERROR: The filtration value excedes the maximum T')
        return

    # Lets compute the filtration of binary images

    # If not scale value a for the filtration is provided, well use the average pixel value in the gray scale image.
    if a==None:
        a = np.mean(np.mean(img_grey))

    # Initialize array to store filtration
    F = np.zeros((img_grey.shape[0], img_grey.shape[1], len(filtration)))

    for i in range(len(filtration)):
        tresh = a*(1 - filtration[i]/T)
        
        temp = np.zeros((img_grey.shape[0], img_grey.shape[1]))

        temp[img_grey <= tresh] = 255

        F[:,:,i] = temp

    # Now we need to compute the lifetime of each point

    lifetimes = np.zeros((img_grey.shape[0], img_grey.shape[1]))

    S_1 = np.column_stack((np.where(F[:,:,1] == 255)[0], np.where(F[:,:,1] == 255)[1]))

    for j in range(S_1.shape[0]):
        if np.where(F[S_1[j,0], S_1[j,1], :] == 0)[0].size > 0:
            lifetimes[S_1[j,0], S_1[j,1]] = np.where(F[S_1[j,0], S_1[j,1], :] == 0)[0][0]-1
        else:
            lifetimes[S_1[j,0], S_1[j,1]] = len(F[S_1[j,0], S_1[j,1], :])
        
    # fig, ax = plt.subplots(1,1, figsize=(10,10))

    # im = ax.imshow(lifetimes)
    # fig.colorbar(im, ax=ax)

    # plt.savefig('lifetimes.png')

    return F, lifetimes

def getSegmentation(F):
    ''' 
    This function computes the step in the filtration of binary images that we will use later onto find a suitable mask.

    :type F: numpy.array
    :param F: 3D array representing the filtration of binary images. The dimensions correspond to height x width x step_in_filtration.

    :retrun: numpy.array - Binary image that will be used to compute the mask.3
    '''

    if F.ndim != 3:
        print('ERROR: The parameter F must be a 3D-array.')
        return

    
    prev_num_connected_componnets = img.shape[0]*img.shape[1]
    T2 = 0
    for i in range(F.shape[2]):
        
        # Compute 0-dim persistence
        dgm = lower_star_img(F[:,:,i])
        # We need to clear out results since nany point with persistence (-0,infy) is point that never was part of the diagram.
        # ind_of_zeros = np.where(dgm[:,0] == 0)[0]
        # dgm = np.delete(dgm, ind_of_zeros, axis=0)

        # The stopping condition comes whenever S_{t+1} has more connected components than S_t
        
        if dgm.shape[0] <= prev_num_connected_componnets:
            
            prev_num_connected_componnets = dgm.shape[0]        
        else:
            
            T2 = i-1
            break
            
    T1 = int(1 + np.floor(T2/4))

    selected = F[:,:,T1]

    return selected

def getMask(img, lifetime):	
    ''' 
    This function computes mas of an image, based on a binary image img and with the lifetime of each pixel in the filtration.

    To accomplish this goal we compute the life score of each connected component C given by

    .. math::
        LS(C) = \frac{ (1 + d(C,o))^3 \sum\limits_{(x,y)\in C} L(x,y) }{ (1 + d(C,b)^3 }

    where $d(C,o)$ is the distance from the component $C$ to the midpoint $o$ and $d(C,d)$ is the distance form the component $C$ to the boundary of the image.

    :type img: numpy.array
    :param F: 2D array binary image.

    :type lifetime: numpy.array
    :param lifetime: 2D array that contains the lifetime for each pixel in a filtration if binary images.

    :retrun: numpy.array - Mask for the image.
    '''

    # This package return the connected components of a binary image pretty fast.
    labeled_image, num_objects = skimage.measure.label(img, connectivity=1, return_num=True)

    # compute mid point of the image
    mid_point = np.array([int(img.shape[0]/2), int(img.shape[1]/2)]).reshape(1,2)

    # find boundary of the image
    top = np.column_stack((np.zeros(img.shape[1]), np.arange(0, img.shape[1])))

    botom = np.column_stack(((img.shape[0]-1)*np.ones(img.shape[1]), np.arange(0, img.shape[1])))

    left = np.column_stack((np.arange(1, img.shape[0]-1), np.zeros(img.shape[0]-2)))

    right = np.column_stack((np.arange(1, img.shape[0]-1), (img.shape[1]-1)*np.ones(img.shape[0]-2)))

    boudnary = np.row_stack((top, botom, left, right))

    LC = []
    for i in range(1,num_objects+1):
        # Points in a given connected component.
        C = np.column_stack((np.where(labeled_image == i)[0], np.where(labeled_image == i)[1]))

        # distance from connected component C to midpoint
        matrix_distance_C_midpont = cdist(mid_point, C, 'minkowski', p=1)

        distance_C_midpont = np.min(matrix_distance_C_midpont)        

        # distance from connected component C to boundary
        matrix_distance_C_boundary = cdist(mid_point, C,'minkowski', p=1)

        distance_C_boundary = max(np.max(np.min(matrix_distance_C_boundary, axis=0)), np.max(np.min(matrix_distance_C_boundary, axis=1)))        

        # Compute Life Score for the current connected component
        LC.append((np.power(1 + distance_C_boundary, 3)*np.sum(lifetime[labeled_image == i]))/np.power(1 + distance_C_midpont, 3))
    
    mean_LC = np.mean(LC)

    choosen_componenets = np.arange(1,num_objects+1)[LC > mean_LC]

    clean_image = np.zeros((img.shape[0], img.shape[1]))
    for j in choosen_componenets:
        clean_image[labeled_image == j] = 255

    # fig, ax = plt.subplots(1,1, figsize=(10,10))

    # ax.imshow(clean_image, cmap='gray')

    # plt.savefig('segmented_clean.png')

    chull = skimage.img_as_ubyte(convex_hull_image(clean_image))

    return chull

def compute_persistence(diagram):
    f_lower_star = d.fill_freudenthal(diagram, reverse = False)
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
            b = 5000
        else:
            b = death
        t = [birth, b]
        jth_pt.append(t)
    dim_list.append(np.array(jth_pt))

# -------------- IMPORT DATA --------------------------------------------------

# Decompres data set, this line is specific for linux!
### Need to add to check if there is a .gitignore already
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

if rank == 0:
    if os.path.exists("data/"):
        print("Data already unzipped")
        os.system('touch data/.gitignore')
        os.system('echo "*" >> data/.gitignore')
    else:
        os.system('tar -xzvf ham10000_images.tar')
        os.system('mv images data')
        os.system('touch data/.gitignore')
        os.system('echo "*" >> data/.gitignore')

# Load after file is unzipped
with open("../config.json") as json_config_file:
    filepaths= json.load(json_config_file)
fp = filepaths['fp']
data = fp + filepaths['ham10000data']
pers = fp + filepaths['ham10000pers']

# -----------------------------------------------------------------------------
# Load labels
# 0 - MEL: “Melanoma” diagnosis confidence
# 1 - NV: “Melanocytic nevus” diagnosis confidence
# 2 - BCC: “Basal cell carcinoma” diagnosis confidence
# 3 - AKIEC: “Actinic keratosis / Bowen’s disease (intraepithelial carcinoma)” diagnosis confidence
# 4 - BKL: “Benign keratosis (solar lentigo / seborrheic keratosis / lichen planus-like keratosis)” diagnosis confidence
# 5 - DF: “Dermatofibroma” diagnosis confidence
# 6 - VASC: “Vascular lesion” diagnosis confidence

image_ids = []
original_labels = []

with open("HAM10000_metadata.csv", newline='') as csvfile:
    csvreader = csv.reader(csvfile, delimiter=',') #csv reader
    next(csvreader) # skip first row / header
    for row in csvreader:
        image_ids.append(row[1])
        original_labels.append(row[2])

original_labels = [0 if x=='mel' else x for x in original_labels]
original_labels = [1 if x=='nv' else x for x in original_labels]
original_labels = [2 if x=='bcc' else x for x in original_labels]
original_labels = [3 if x=='akiec' else x for x in original_labels]
original_labels = [4 if x=='bkl' else x for x in original_labels]
original_labels = [5 if x=='df' else x for x in original_labels]
original_labels = [6 if x=='vasc' else x for x in original_labels]

# -----------------------------------------------------------------------------

l = os.listdir(data)
l.remove('.gitignore')
try:
    l.remove('.DS_Store')
except:
    print("No .DS_Store file to remove")

dgm_R_rgb_0 = []
dgm_G_rgb_0 = []
dgm_B_rgb_0 = []

dgm_R_rgb_1 = []
dgm_G_rgb_1 = []
dgm_B_rgb_1 = []

dgm_H_hsv_0 = []
dgm_S_hsv_0 = []
dgm_V_hsv_0 = []

dgm_H_hsv_1 = []
dgm_S_hsv_1 = []
dgm_V_hsv_1 = []

dgm_X_xyz_0 = []
dgm_Y_xyz_0 = []
dgm_Z_xyz_0 = []

dgm_X_xyz_1 = []
dgm_Y_xyz_1 = []
dgm_Z_xyz_1 = []

labels = []

plt.ioff()

start = sys.argv[1]
start = int(start)
end = sys.argv[2]
end = int(end)

if rank == 0:
    index = [i for i in range(start,end)]
    index = np.array_split(index,size)
else: 
    index = None
index = comm.scatter(index)

#for file in l:
for i in index:
    file = l[i]
    print(rank)
    print(l[i])
    filename = os.fsdecode(data + file)

    img = mpimg.imread(filename)

    # This function returns a filtration of binary images as a 3D-array.
    Fill, lifetimes = getImageFiltration(img, T=50, filtration=np.linspace(0,50,20))
    segmented_image = getSegmentation(Fill)
    mask = getMask(segmented_image, lifetimes)

    # Apply mask to image

    masked_img = cv2.bitwise_and(img, img, mask = mask)

    # RGB
    R = compute_persistence(np.array(-masked_img[:,:,0], dtype=float))
    append_dim_list(R[0], dgm_R_rgb_0)
    append_dim_list(R[1], dgm_R_rgb_1)
    
    G = compute_persistence(np.array(-masked_img[:,:,1], dtype=float))
    append_dim_list(G[0], dgm_G_rgb_0)
    append_dim_list(G[1], dgm_G_rgb_1)
    
    B = compute_persistence(np.array(-masked_img[:,:,2], dtype=float)) 
    append_dim_list(B[0], dgm_B_rgb_0)
    append_dim_list(B[1], dgm_B_rgb_1)

    # HSV
    masked_img = skimage.color.rgb2hsv(masked_img)

    H = compute_persistence(np.array(-masked_img[:,:,0], dtype=float)) 
    append_dim_list(H[0], dgm_H_hsv_0)
    append_dim_list(H[1], dgm_H_hsv_1)
    
    S = compute_persistence(np.array(-masked_img[:,:,1], dtype=float)) 
    append_dim_list(S[0], dgm_S_hsv_0)
    append_dim_list(S[1], dgm_S_hsv_1)
    
    V = compute_persistence(np.array(-masked_img[:,:,2], dtype=float)) 
    append_dim_list(V[0], dgm_V_hsv_0)
    append_dim_list(V[1], dgm_V_hsv_1)

    # XYZ  

    masked_img = skimage.color.rgb2xyz(masked_img)

    X = compute_persistence(np.array(-masked_img[:,:,0], dtype=float)) 
    append_dim_list(X[0], dgm_X_xyz_0)
    append_dim_list(X[1], dgm_X_xyz_1)
    
    Y = compute_persistence(np.array(-masked_img[:,:,1], dtype=float)) 
    append_dim_list(Y[0], dgm_Y_xyz_0)
    append_dim_list(Y[1], dgm_Y_xyz_1)
    
    Z = compute_persistence(np.array(-masked_img[:,:,2], dtype=float)) 
    append_dim_list(Z[0], dgm_Z_xyz_0)
    append_dim_list(Z[1], dgm_Z_xyz_1)

    # Now lets find the correct label

    labels.append(original_labels[image_ids.index(file[:-4])])

    print('Processed file ', filename)

diagrams = pd.DataFrame(index=range(len(index)))

diagrams['dgm_R_rgb_0'] = dgm_R_rgb_0
diagrams['dgm_G_rgb_0'] = dgm_G_rgb_0
diagrams['dgm_B_rgb_0'] = dgm_B_rgb_0

diagrams['dgm_R_rgb_1'] = dgm_R_rgb_1
diagrams['dgm_G_rgb_1'] = dgm_G_rgb_1
diagrams['dgm_B_rgb_1'] = dgm_B_rgb_1

diagrams['dgm_H_hsv_0'] = dgm_H_hsv_0
diagrams['dgm_S_hsv_0'] = dgm_S_hsv_0
diagrams['dgm_V_hsv_0'] = dgm_V_hsv_0

diagrams['dgm_H_hsv_1'] = dgm_H_hsv_1
diagrams['dgm_S_hsv_1'] = dgm_S_hsv_1
diagrams['dgm_V_hsv_1'] = dgm_V_hsv_1

diagrams['dgm_X_xyz_0'] = dgm_X_xyz_0
diagrams['dgm_Y_xyz_0'] = dgm_Y_xyz_0
diagrams['dgm_Z_xyz_0'] = dgm_Z_xyz_0

diagrams['dgm_X_xyz_1'] = dgm_X_xyz_1
diagrams['dgm_Y_xyz_1'] = dgm_Y_xyz_1
diagrams['dgm_Z_xyz_1'] = dgm_Z_xyz_1

diagrams['labels'] = labels

diagrams = comm.gather(diagrams, root=0)
if rank == 0:
    dataFrame = diagrams[0]
    for i in range(1,len(diagrams)):
        dataFrame=dataFrame.append(diagrams[i], ignore_index=True)

    # Save with pickle
    if os.path.exists(pers):
        os.system('touch ' + pers + '/.gitignore')
        os.system('echo "*" >> ' + pers + '/.gitignore')
    else:
        os.system('mkdir ' + pers)
        os.system('touch ' + pers + '/.gitignore')
        os.system('echo "*" >> ' + pers + '/.gitignore')
    diagram_name = 'diagrams_' + str(start) + '_' + str(end) + '.pickle'
    with open(pers + diagram_name, 'wb') as handle:
        pickle.dump(dataFrame, handle, protocol=pickle.HIGHEST_PROTOCOL)
