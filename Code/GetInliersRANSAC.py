#/usr/bin/evn python

"""
RBE/CS Spring 2023: Classical and Deep Learning Approaches for
Geometric Computer Vision
Project 1: Buildings built in minutes - SfM and NeRF Phase 1 

Author(s):
Uthiralakshmi Sivaraman (usivaraman@wpi.edu)
Worcester Polytechnic Institute
"""

# imports 

import cv2
import numpy as np
import matplotlib.pyplot as plt


from EstimateFundamentalMatrix import *

# We use SIFT features to find features in each paird of images and use RANSAC to remove outliers, we apply RANSAC on Fundamental Matrix 

def GetInliersRANSAC(pts1,pts2,NMAX,match_threshold,idx):


    inliers_threshold = 0
    inliers_indices = []
    f_inliers = None
    
    for _ in range(NMAX):

        number_matches = pts1.shape[0]
        random_8Indices = np.random.choice(number_matches,8)

        x1 = pts1[random_8Indices,:]
        x2 = pts2[random_8Indices,:]
        F = EstimateFundamentalMatrix(x1,x2)
        indices = []    


        

        indices =[]
        if F is not None :
            # print("Estimated F matrix ",F)
            for i in range(number_matches):
                x11 = np.array([ pts1[i,0],pts1[i,1],1])
                x22 = np.array([ pts2[i,0],pts2[i,1],1]).T
                # print("x1 :",x1)
                # print("x2 :",x2)
                error = np.abs(np.dot(x22,np.dot(F,x11))) #emphasising the condition that #x2.TFx1 = 0 Epipolar Constrainst 
                if(error<match_threshold):
                       indices.append(idx[i])
                       
        
        if len(indices) > inliers_threshold:
            inliers_threshold = len(indices)
            inliers_indices = indices
            f_inliers = F  

    return F, inliers_indices