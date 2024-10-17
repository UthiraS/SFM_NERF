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
import os
import sys
from mpl_toolkits import mplot3d
from scipy.spatial.transform import Rotation 
import scipy

def readImages(ImagePath):

    
    if os.path.exists(ImagePath): 
        image_list = os.listdir(ImagePath)
        image_list.sort()
    else:
        raise Exception ("Directory doesn't exist")
    images_path = []
    for i in range(len(image_list)):
        image_path = os.path.join(ImagePath,image_list[i])
        images_path.append(image_path)

    images = [cv2.imread(i) for i in images_path]
    
    return images


def getRotation(Q, type_ = 'q'):
    if type_ == 'q':
        R = Rotation.from_quat(Q)
        return R.as_matrix()
    elif type_ == 'e':
        R = Rotation.from_rotvec(Q)
        return R.as_matrix()

def project(points_3d, camera_params, K):
    def projectPoint_(R, C, pt3D, K):
        P2 = np.dot(K, np.dot(R, np.hstack((np.identity(3), -C.reshape(3,1)))))
        x3D_4 = np.hstack((pt3D, 1))
        x_proj = np.dot(P2, x3D_4.T)
        x_proj /= x_proj[-1]
        return x_proj

    x_proj = []
    for i in range(len(camera_params)):
        
        R = getRotation(camera_params[i, :3], 'e')
        C = camera_params[i, 3:].reshape(3,1)
        pt3D = points_3d[i]
        pt_proj = projectPoint_(R, C, pt3D, K)[:2]
        x_proj.append(pt_proj)    
    return np.array(x_proj)

def rotate(points, rot_vecs):
    """Rotate points by given rotation vectors.
    
    Rodrigues' rotation formula is used.
    """
    theta = np.linalg.norm(rot_vecs, axis=1)[:, np.newaxis]
    with np.errstate(invalid='ignore'):
        v = rot_vecs / theta
        v = np.nan_to_num(v)
    dot = np.sum(points * v, axis=1)[:, np.newaxis]
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)

    return cos_theta * points + sin_theta * np.cross(v, points) + dot * (1 - cos_theta) * v

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


