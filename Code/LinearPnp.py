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


# After we have found X points and R and C from linear triangulation, non linear triangulation and disambiguate camera pose
# we add the third camera view point using Perspective N points

def LinearPnp(Xpoints,xpoints,K):


    
    A =[]

    for i,(xp,Xp) in enumerate(zip(xpoints,Xpoints)):

        # print("xp ",xp)
        # print("Xp ",Xp)

        x,y = xp
        X,Y,Z = Xp



        A.append([X,Y,Z,1,0,0,0,0,-x*X,-x*Y,-x*Z, -x])
        A.append([0,0,0,0,X,Y,Z,1,-y*X,-y*Y,-y*Z,-y])

    # print("A matrix obtained for PNP from xpoints and Xpoints :",A)

    u,s,vh = np.linalg.svd(A)
    p = vh[np.argmin(s),:]

    # print("p :",p)
    # print("p.shape :",p.shape)

    #reshaping the p matrix obtained
    p =p.reshape((len(p),1))
    p = p.reshape((3,4))

    # print("p.shape :",p)
    
    K_Inv = np.linalg.inv(K)

    # print("K_Inv.shape :",K_Inv)


    U,D,VT = np.linalg.svd(K_Inv*p[0:3,0:3])

    # print(" U :",U)
    # print(" D :",D)
    # print(" VT :",VT)

    R = U.dot(VT)
    T = K_Inv.dot(p[:,3])/D[0]


    if np.linalg.det(R) == -1:

        R=-R
        T =-T

    return R,T



