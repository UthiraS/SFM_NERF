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
from scipy.spatial.transform import Rotation 
import scipy.optimize as optimize


#Given N≥6 3D-2D correspondences, X↔x, and linearly estimated camera pose, (C,R), refine the camera pose that minimizes reprojection error using Non Linear PNP

def NonLinearPnp(K, pts, x3D, R0, C0):

   
    Q = Rotation.from_matrix(R0)
    Q = Q.as_quat()
    X0 = [Q[0] ,Q[1],Q[2],Q[3], C0[0], C0[1], C0[2]] 
    optimized_params = optimize.least_squares(
        fun = Loss,
        x0=X0,
        method="trf",
        args=[x3D, pts, K])
    X1 = optimized_params.x
    Q = X1[:4]
    C = X1[4:]
    R = Rotation.from_quat(Q)
    R = R.as_matrix()
    return R, C

def Loss(X0,x3D, pts, K):


    Q, C = X0[:4], X0[4:].reshape(-1,1)
    R = Rotation.from_quat(Q)
    R = R.as_matrix()
    C = np.reshape(C, (3, 1))        
    I = np.identity(3)
    P = np.dot(K, np.dot(R, np.hstack((I, -C))))

    # print("P :",P)
    
    Error = []
    for X, pt in zip(x3D, pts):

        p_1T, p_2T, p_3T = P# rows of P
        p_1T, p_2T, p_3T = p_1T.reshape(1,-1), p_2T.reshape(1,-1), p_3T.reshape(1,-1)

        X = X.reshape(1,-1)
        X = np.hstack((X, np.ones((X.shape[0], 1)))).reshape(-1,1) 
       
        u, v = pt[0], pt[1]
        u_proj = np.divide(p_1T.dot(X) , p_3T.dot(X))
        v_proj =  np.divide(p_2T.dot(X) , p_3T.dot(X))

        E = np.square(v - v_proj) + np.square(u - u_proj)

        Error.append(E)

    sumError = np.mean(np.array(Error).squeeze())
    return sumError



