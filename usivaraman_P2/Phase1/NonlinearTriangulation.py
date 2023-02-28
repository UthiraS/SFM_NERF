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
import scipy.optimize as optimize


# After we got two camera poses and World Points X found based on linear triangulation and cheriality condition, 
#we compute the geometric error between the given point and reprojected point and minimize the error using optimization function from scipy.
def NonlinearTriangulation(K,R1,C1,R2,C2,x1, x2, X):


    I = np.identity(3)
    C1 = C1.reshape((3,1))
    C2 = C2.reshape((3,1))
    P1 = np.dot(K, np.dot(R1, np.hstack((I, -C1)))) 
    P2 = np.dot(K, np.dot(R2, np.hstack((I, -C2))))


    X_optimized =[]

    for i, Xpoint in enumerate(X):
        optimized_params = optimize.least_squares(fun=ErrorFunction, x0=Xpoint, method="trf", args=[x1[i], x2[i], P1, P2])
        X_opt = optimized_params.x
        X_optimized.append(X_opt)
    return np.array(X_optimized)




def ErrorFunction(X,x1,x2,P1,P2):

    #Calculating reprojection error for world Point and Camera 1

    u1,v1 = x1[0], x1[1]
    u1_proj = P1[0].dot(X)/P1[2].dot(X)
    v1_proj =  P1[1].dot(X)/P1[2].dot(X)
    E1= np.square(v1 - v1_proj) + np.square(u1 - u1_proj)


    # print("Error 1 :",E1)

    #Calculating reprojection error for world Point and Camera 2
       
    u2,v2 = x2[0], x2[1]
    u2_proj = P2[0].dot(X)/P2[2].dot(X)
    v2_proj =  P2[1].dot(X)/P2[2].dot(X)
    E2= np.square(v2 - v2_proj) + np.square(u2 - u2_proj)

    # print("Error 2 :",E2)

    #combining the two errors 

    total_error = E1+ E2

    # print("Total Error:",total_error)
    return total_error






    