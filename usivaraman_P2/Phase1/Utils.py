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



def projectionMatrix(R,C,K):
    C = np.reshape(C,(3,1))
    I = np.identity(3)
    P = np.dot(K,np.dot(R,np.hstack((I,-C))))
    return P

def ReProjectionError(X,pt1, pt2, R1, C1, R2, C2, K):
    p1 = projectionMatrix(R1, C1, K)
    p2 = projectionMatrix(R2, C2, K)

    # print("This i s p1",p1,p1.shape)
    # print("This i s p2",p2,p2.shape)
    p1_1T, p1_2T, p1_3T = p1
    # print(p1_1T.shape, p1_2T.shape, p1_3T.shape )
    p1_1T, p1_2T, p1_3T = p1_1T.reshape(1,4), p1_2T.reshape(1,4), p1_3T.reshape(1,4)

    p2_1T, p2_2T, p2_3T = p2
    p2_1T, p2_2T, p2_3T = p2_1T.reshape(1,4), p2_2T.reshape(1,4), p2_3T.reshape(1,4)

    X = X.reshape(4,1)

    "Reprojection error w.r.t 1st Ref camera points"
    u1, v1 = pt1[0], pt1[1]
    # print(u1,v1)
    # print(p1_1T.shape,(p1_3T.shape),X.shape)
    u1_projection = np.divide(p1_1T.dot(X), p1_3T.dot(X))
    v1_projection = np.divide(p1_2T.dot(X), p1_3T.dot(X))
    err1 = np.square(v1 - v1_projection) + np.square(u1 - u1_projection)

    "Reprojection error w.r.t 2nd Ref camera points"
    u2, v2 = pt2[0], pt2[1]
    u2_projection = np.divide(p2_1T.dot(X), p2_3T.dot(X))
    v2_projection = np.divide(p2_2T.dot(X), p2_3T.dot(X))
    err2 = np.square(v2 - v2_projection) + np.square(u2 - u2_projection)

    return err1, err2

def reprojectionErrorPnP(x3D, pts, K, R, C):
    P = projectionMatrix(R,C,K)
    # print("P :",P)
    
    Error = []
    for X, pt in zip(x3D, pts):

        p_1T, p_2T, p_3T = P# rows of P
        p_1T, p_2T, p_3T = p_1T.reshape(1,-1), p_2T.reshape(1,-1), p_3T.reshape(1,-1)
        X = X.reshape(1,-1)
        X = np.hstack((X, np.ones((X.shape[0], 1)))).reshape(-1,1) # make X it a column of homogenous vector
        ## reprojection error for reference camera points 
        u, v = pt[0], pt[1]
        u_proj = np.divide(p_1T.dot(X) , p_3T.dot(X))
        v_proj =  np.divide(p_2T.dot(X) , p_3T.dot(X))

        E = np.square(v - v_proj) + np.square(u - u_proj)

        Error.append(E)

    mean_error = np.mean(np.array(Error).squeeze())
    return mean_error


