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


def DrawCameras(R, C,plt,ax,label):
    angle = Rotation.from_matrix(R).as_euler("XYZ")
    angle = np.rad2deg(angle)
    plt.plot(C[0],C[2],marker=(3, 0, int(angle[1])),markersize=15,linestyle='None') 
    corr = -0.1
    ax.annotate(label,xy=(C[0]+corr,C[2]+corr))



def feature_matching(img1, img2, pts_1,pts_2,matches,name):
  
    points1=[]
    for i in range(len(pts_1)):
        points1.append(cv2.KeyPoint(int(pts_1[i][0]), int(pts_1[i][1]), 3))
    points2=[]
    for i in range(len(pts_2)):
        points2.append(cv2.KeyPoint(int(pts_2[i][0]), int(pts_2[i][1]), 3))

    pairs_idx =  [(i,i) for i,j in enumerate(matches)]
    m = []
    for i in range(len(pairs_idx)):
        m.append(cv2.DMatch(int(pairs_idx[i][0]), int(pairs_idx[i][1]), 2))
    matches = m
	# i= 0
	# for distance in distances:
	# 	matches.append(cv.DMatch(i,i,2))
	# 	i = i+1

    ret = np.array([])
    out = cv2.drawMatches(img1=img1,
        keypoints1=points1,
        img2=img2,
        keypoints2=points2,
        matches1to2=matches,outImg = None,flags =2)
    plt.imshow(out)
    plt.savefig("../Phase1/Data/IntermediateOutputImages/"+name+".jpg")
   
    plt.show()
    plt.close()

def plot_camdis(X):

    plt.figure("camera disambiguation")
    # plt.set(xlim=(-30, 30), ylim=(-30,30))
    plt.xlim([-40, 40])
    plt.ylim([-30, 30])
    colors = ['red','brown','greenyellow','teal']
    for color, X_c in zip(colors, X):
        plt.scatter(X_c[:,0],X_c[:,2],color=color,marker='.')
    plt.savefig("../Phase1/Data/IntermediateOutputImages/DisambiguateCameraPose.jpg")
   
    plt.show()
    plt.close()

def plot_LTNLT(Xpose,X_optimized,Rpose,Cpose,R1_,C1_):


    fig = plt.figure("LT, NLT")
    ax = fig.add_subplot()
    plt.xlim([-30, 30])
    plt.ylim([-10, 30])
    plt.scatter(Xpose[:,0], Xpose[:,2],c="b",s=1,label="Linear Triangulation")
    plt.scatter(X_optimized[:,0], X_optimized[:,2],c="r",s=1,label="NonLinear_Triangulation")
    DrawCameras(R1_,C1_,plt,ax,"1") #Draw 1st camera 
    DrawCameras(Rpose,Cpose,plt,ax,"2")  # Draw 2nd camera
    plt.xlim([-20, 20])
    plt.ylim([-15, 15])
    plt.legend()
    plt.savefig('../Phase1/Data/IntermediateOutputImages/'+'LTNLT.png')
    plt.show()
    plt.close()

def final_plot(X,C_set_,R_set_):

    x = X[:,0]
    y = X[:,1]
    z = X[:,2]

    
    
    fig = plt.figure(figsize = (10,10))
    plt.xlim(-4,6)
    plt.ylim(-2,12)
    plt.scatter(x,z,marker='.',linewidths=0.5, color = 'blue')
    for i in range(0, len(C_set_)):
        R1 = Rotation.from_matrix(R_set_[i]).as_euler("XYZ")
    
        R1 = np.rad2deg(R1)
        plt.plot(C_set_[i][0],C_set_[i][2], marker=(3,0, int(R1[1])), markersize=15, linestyle='None')

    plt.savefig('../Phase1/Data/IntermediateOutputImages/'+'2D.png')
    plt.show()
    plt.close()
