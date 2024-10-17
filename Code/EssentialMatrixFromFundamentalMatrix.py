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

def EssentialMatrixFromFundamentalMatrix(K,F):



    E = K.T.dot(F).dot(K)

    # Emphasising s = [1,1,0] 
    u, s, vt = np.linalg.svd(E)
    # s = np.diag(s)
    s = [1,1,0]                 
    E = np.dot(u, np.dot(np.diag(s), vt))

    print(" E matrix after Emphasising s = [1,1,0]  :",E )
    print("Rank of  E matrix :",np.linalg.matrix_rank(E))


    return E 

