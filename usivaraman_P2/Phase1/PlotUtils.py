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