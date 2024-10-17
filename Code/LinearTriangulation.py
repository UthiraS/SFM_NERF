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

# After we have found that the 4 different camera poses R1,C1,R2,C2 and we have information of image points, we find the World Points X using Linear Triangulation, similar to the concept of linear trilateration concept used in GPS 

def LinearTriangulation(K,R1,C1,R2,C2,xpts1,xpts2):


    # print("Points input to Linear Triangulation:")
    # print("pts1 :",xpts1)
    # print("pts2 :",xpts2)
    # print("pts1.shape :",xpts1.shape)
    # print("pts2.shape :",xpts2.shape)
    xpts1 = np.hstack((xpts1, np.ones((xpts1.shape[0], 1)))) #Appending one to x1 and x2 
    xpts2 = np.hstack((xpts2, np.ones((xpts2.shape[0], 1))))

    C1 = C1.reshape((3,1))
    C2 = C2.reshape((3,1))
    I = np.identity(3)
    P1 = np.dot(K, np.dot(R1, np.hstack((I, -C1))))  # the P is written this way but not as P = K[R T] because the C and R here means the rotation and translation between cameras 
    P2 = np.dot(K, np.dot(R2, np.hstack((I, -C2))))
    X=[]
    for i, (xp1,xp2) in enumerate(zip(xpts1,xpts2)):

        Sxpt1 = np.array([[0, -xp1[2], xp1[1]], [xp1[2], 0, -xp1[0]], [-xp1[1], xp1[0], 0]])
        Sxpt2 = np.array([[0, -xp2[2], xp2[1]], [xp2[2], 0, -xp2[0]], [-xp2[1], xp2[0], 0]])

        # print("Skew Symmetric Matrix formed out of pts :")
        # print(" Sxpt1 :",Sxpt1)
        # print(" Sxpt2 :",Sxpt2)

        S1 = Sxpt1.dot(P1)
        S2 = Sxpt2.dot(P2)

        # print("Dot Product of skew symmetric matrix of small x points and projection matrix  :")
        # print(" S1 :",S1)
        # print(" S2 :",S2)

        A = np.vstack((S1,S2))

        # print("A matrix :",A)

        u,s, vh = np.linalg.svd(A)
        x = vh[np.argmin(s),:] 
        # print("X  :",x)
        x = x/x[3]  
        X.append(x)
    X = np.vstack(X) 
    return X
















