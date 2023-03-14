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
from LinearPnp import *

#PnP is prone to error as there are outliers in the given set of point correspondences, 
# to make camera pose more robust to outliers, given N≥6 3D-2D correspondences, we use RANSAC to filter outliers

def PnPRANSAC(Xpts,xpts,K, NMAX,thresh):


    inliers_thresh =0
    num_WorldPoints = Xpts.shape[0]
    R_selected = None
    C_selected = None
    thresh = 200
    for i in range(NMAX):

        
        random_6Indices = np.random.choice(num_WorldPoints,6)
        X_batch, x_batch = Xpts[random_6Indices], xpts[random_6Indices]

        # print("X_batch.shape",X_batch.shape)
        # print("x_batch.shape",x_batch.shape)
        
        R,C = LinearPnp(X_batch, x_batch,K)

        # print("R :",R)
        # print("C :",C)

        indices =[]
        if R is not None:

            for i in range(num_WorldPoints):
                
                xp= xpts[i]
                Xp = Xpts[i]
                u,v = xp
                pts = Xp.reshape(1,-1)
                X = np.hstack((pts, np.ones((pts.shape[0],1))))
                X = X.reshape(4,1)
                C = C.reshape(-1,1)


                I = np.identity(3)
                C = np.reshape(C, (3, 1)) 
                P = np.dot(K, np.dot(R, np.hstack((I, -C))))
               
                p1, p2, p3 = P
                p1, p2, p3 = p1.reshape(1,4), p2.reshape(1,4), p3.reshape(1,4)


                u_proj = np.divide(p1.dot(X), p3.dot(X))
                v_proj = np.divide(p2.dot(X), p3.dot(X))

                x_proj = np.hstack((u_proj, v_proj))
                x = np.hstack((u,v))
                error = np.linalg.norm(x-x_proj)

                # print("error :",error)

                if error < thresh:
                    indices.append(i)

        if len(indices) > inliers_thresh:
            inliers_thresh = len(indices)
            R_selected = R
            C_selected = C

    return R_selected, C_selected