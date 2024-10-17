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


# From the Essential Matrix, We obtain the value of Rotation and Translation values, Mathematically 4 solutions exist

def ExtractCameraPose(E):

    U, S, Vt = np.linalg.svd(E)
    W = np.array([[0, -1, 0],
                  [1, 0, 0],
                  [0, 0, 1]])
    
    # print("U  :",U)
    # print("S  :",S)
    # print("Vt  :",Vt)
    # print("W  :",W)

    C1 = U[:,2]
    C2 = -U[:,2]
    C3 = U[:,2]
    C4 = -U[:,2]



    R1 = U.dot(W).dot(Vt)
    R2 = U.dot(W).dot(Vt)
    R3 = U.dot(W).dot(Vt.T)
    R4 = U.dot(W).dot(Vt.T)

    # print(" C1, R1 :", C1,R1)
    # print(" C2, R2 :", C2,R2)
    # print(" C3, R3 :", C3,R3)
    # print(" C4, R4 :", C4,R4)
    C =[]
    R =[]
    Camera_Poses = [[C1,R1],[C2,R2],[C3,R3],[C4,R4]]

    for i,cameraPose in enumerate(Camera_Poses):

        if np.linalg.det(cameraPose[1]) <0:

            print("Camera Pose :",cameraPose)

            cameraPose[0] = -cameraPose[0]
            cameraPose[1] = -cameraPose[1]
            Camera_Poses[i] = cameraPose

        C.append(cameraPose[0])
        R.append(cameraPose[1])


    # print(" C1, R1 :", C1,R1)
    # print(" C2, R2 :", C2,R2)
    # print(" C3, R3 :", C3,R3)
    # print(" C4, R4 :", C4,R4)

    # C = [C1,C2,C3,C4]
    # R = [R1,R2,R3,R4]

    return C,R



            


            





