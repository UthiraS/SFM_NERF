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


# Given a pair of 8 points, we obtain the Fundamental Matrix
# We first compute A matrix from AF =0 and do SVD to compute to F
# Based on the F obtained, we emphasise the condition that rank of F is 2 so that we avoid the noise in F

def EstimateFundamentalMatrix(pts1,pts2):

    # Comverting the match points into (x1,y1) and (x2,y2) 

   
    A =[]

    for index in range(len(pts1)):

        x1 = pts1[index][0]
        y1 = pts1[index][1]
        x2 = pts2[index][0]
        y2 = pts2[index][1]

        A.append([x1*x2, y1*x2, x2, x1*y2, y1*y2, y2, x1, y1, 1])

    # print(" A matrix :", A)

    U,S,VT = np.linalg.svd(A,full_matrices=True)
    # F=VT[np.argmin(S)]

    F1 = VT.T[:,-1]

    # print(" F matrix :", F)
    # print("Rank of  F matrix :",np.linalg.matrix_rank(F))
    # print(" F1 matrix :", F1)
    
    F1 = F1.reshape(3,3)
    # print(" F1 matrix reshaped :",F1 )
    # print("Rank of  F1 matrix :",np.linalg.matrix_rank(F1))

    # Emphasising Rank 2
    u, s, vt = np.linalg.svd(F1)
    s = np.diag(s)
    s[2,2] = 0                     
    F1 = np.dot(u, np.dot(s, vt))

    # print(" F1 matrix after Rank 2 :",F1 )
    # print("Rank of  F1 matrix :",np.linalg.matrix_rank(F1))


    return F1


    


     

    


    

