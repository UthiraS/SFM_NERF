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


# From 4 possible camera poses, we find the pose in which the world points lie in front of both the cameras

def DisambiguateCameraPose(RSet,CSet,XValues):

    best_index = 0
    max_positive_depths = 0
    for i in range(len(RSet)):
        R, C= RSet[i], CSet[i]
        # print("R :",R) 
        # print("R  shape:",R.shape)
        # print("C :",C)
        # print("C  shape:",C.shape)
        

        C = np.transpose(C).reshape(-1,1)
        print(C)
        r3 = R[2].reshape(1,-1)
        counter = 0
        for X in XValues[i]:

            print(X)

            X = X/X[3]       
            X = X[0:3]
            X =X.reshape(-1,1)

            if r3.dot(X-C).T>0 and X[2]>0:
                counter+=1 

        if(counter>max_positive_depths):

            max_positive_depths = counter
            best_index = i
    return RSet[best_index],CSet[best_index],XValues[best_index]
        

         
      
       
        
       


      

