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

import numpy as np
import time
import cv2
from scipy.spatial.transform import Rotation
from scipy.sparse import lil_matrix
import scipy.optimize as optimize
from BuildVisibilityMatrix import *
from Utils import *

#Reference:  Large-Scale bundle Adjustment in Scipy
"https://scipy-cookbook.readthedocs.io/items/bundle_adjustment.html"
#simultaneously refine the 3D point locations and camera poses that were estimated 
# minimize the difference between the observed image points and the corresponding points projected from the estimated 3D
# locations and camera poses



def bundle_adjustment_sparsity(X_found, filtered_feature_flag, nCam):
    
   

    number_of_cam = nCam + 1
    X_index, visiblity_matrix = vis_mat_func(X_found.reshape(-1), filtered_feature_flag, nCam)
    n_observations = np.sum(visiblity_matrix)
    n_points = len(X_index[0])

    m = n_observations * 2
    n = number_of_cam * 6 + n_points * 3  
    A = lil_matrix((m, n), dtype=int)
    

    i = np.arange(n_observations)
    camera_indic = []
    point_indic = []
    h, w = visiblity_matrix.shape
    for i in range(h):
        for j in range(w):
            if visiblity_matrix[i,j] == 1:
                camera_indic.append(j)
                point_indic.append(i)

    camera_indices = np.array(camera_indic).reshape(-1)
    point_indices = np.array(point_indic).reshape(-1)

    # print("camera indices :",camera_indices)
    # print("point indices :",point_indices)
    i = np.arange(n_observations)

    for s in range(6):
        A[2 * i, camera_indices * 6 + s] = 1
        A[2 * i + 1, camera_indices * 6 + s] = 1

    for s in range(3):
        A[2 * i, (nCam)* 6 + point_indices * 3 + s] = 1
        A[2 * i + 1, (nCam) * 6 + point_indices * 3 + s] = 1

    return A    



def loss_fun(x0, nCam, n_points, camera_indices, point_indices, points_2d, K):
    
    number_of_cam = nCam + 1
    camera_params = x0[:number_of_cam * 6].reshape((number_of_cam, 6))
    points_3d = x0[number_of_cam * 6:].reshape((n_points, 3))
    points_proj = project(points_3d[point_indices], camera_params[camera_indices], K)
    error_vec = (points_proj - points_2d).ravel()
    
    return error_vec  


def BundleAdjustment(X_index,visiblity_matrix,X_all,X_found,feature_x,feature_y, filtered_feature_flag, R_set, C_set, K, nCam):

    # Find the 3D points from  X_all for storing the all world points and X_index is where both the image points and 3 points are valid
    
    points_3d = X_all[X_index]


    # Find the 2D points correspondences from visibility matrix 

    pts2D = []
    visible_feature_x = feature_x[X_index]
    visible_feature_y = feature_y[X_index]
    h, w = visiblity_matrix.shape
    for i in range(h):
        for j in range(w):
            if visiblity_matrix[i,j] == 1:
                pt = np.hstack((visible_feature_x[i,j], visible_feature_y[i,j]))
                pts2D.append(pt)
    points_2d = np.array(pts2D).reshape(-1, 2) 


     # Find the visible indices for both camera and  points from visibility matrix 


    camera_indices = []
    point_indices = []
    h, w = visiblity_matrix.shape
    for i in range(h):
        for j in range(w):
            if visiblity_matrix[i,j] == 1:
                camera_indices.append(j)
                point_indices.append(i)

    camera_indices = np.array(camera_indices).reshape(-1)
    points_indices = np.array(point_indices).reshape(-1)

    # Finding Camera Extrinsics for all the cameras

    # Here, we are using Quarternions instead of Rotation Matrix to preserve orthonormality
    RC = []
    for i in range(nCam+1):
        C, R = C_set[i], R_set[i]
        R=  Rotation.from_matrix(R)
        Q = R.as_rotvec()
        RC_ = [Q[0], Q[1], Q[2], C[0], C[1], C[2]]
        RC.append(RC_)
    RC = np.array(RC, dtype=object).reshape(-1,6)

    # 

    x0 = np.hstack((RC.ravel(), points_3d.ravel()))
    n_pts = points_3d.shape[0]

   
    # Building Sparse Matrix 
   
    A = bundle_adjustment_sparsity(X_found,filtered_feature_flag,nCam)
    print("Bundle Adjustment Sparsity :",A)

    # Optimize , Least Squares 
    res = optimize.least_squares(loss_fun,x0,jac_sparsity=A, verbose=2,x_scale='jac', ftol=1e-10, method='trf',
                        args=(nCam, n_pts, camera_indices, points_indices, points_2d,K))



    x1 = res.x
    no_of_cams = nCam + 1
    optim_cam_param = x1[:no_of_cams*6].reshape((no_of_cams,6))
    optim_pts_3d = x1[no_of_cams*6:].reshape((n_pts,3))

    optim_X_all = np.zeros_like(X_all)
    optim_X_all[X_index] = optim_pts_3d

    optim_C_set , optim_R_set = [], []
    for i in range(len(optim_cam_param)):
        R = getRotation(optim_cam_param[i,:3], 'e')
        C = optim_cam_param[i,3:].reshape(3,1)
        optim_C_set.append(C)
        optim_R_set.append(R)

    return optim_R_set, optim_C_set, optim_X_all


# imports 

# import cv2
# import numpy as np
# import matplotlib.pyplot as plt
# from scipy.spatial.transform import Rotation
# from scipy.sparse import lil_matrix
# from scipy.optimize import least_square

# def bundle_adjustment_sparsity(n_cameras, n_points, camera_indices,
#                                point_indices):
#     """Computes Bundle Adustment for the computed SFM results

#     Args:
#         n_cameras (int): Number of Cameras(6)
#         n_points (int): Number of total Points
#         camera_indices (array): indices of visible cameras
#         point_indices (array): indices of visible points

#     Returns:
#         TYPE: Sparse output Adjustment
#     """
#     m = camera_indices.size * 2
#     n = n_cameras * 9 + n_points * 3
#     A = lil_matrix((m, n), dtype=int)

#     i = np.arange(camera_indices.size)
#     for s in range(9):
#         A[2 * i, camera_indices * 9 + s] = 1
#         A[2 * i + 1, camera_indices * 9 + s] = 1

#     for s in range(3):
#         A[2 * i, n_cameras * 9 + point_indices * 3 + s] = 1
#         A[2 * i + 1, n_cameras * 9 + point_indices * 3 + s] = 1

#     return A


# def BundleAdjustment(Cset, Rset, X, K, points_2d, camera_indices, recon_bin,
#                      V_bundle):
  
#     f = K[1, 1]
#     camera_params = []
#     # camera_indices  = np.array(r_indx[1:])
#     point_indices, _ = np.where(recon_bin == 1)
#     V = V_bundle[point_indices, :]
#     points_3d = X[point_indices, :]
#     for C0, R0 in zip(Cset, Rset):
#         q_temp = Rscipy.from_matrix(R0)
#         Q0 = q_temp.as_rotvec()
#         params = [Q0[0], Q0[1], Q0[2], C0[0], C0[1], C0[2], f, 0, 0]
#         camera_params.append(params)

#     camera_params = np.reshape(camera_params, (-1, 9))

#     n_cameras = camera_params.shape[0]

#     assert len(Cset) == n_cameras, "length not matched"

#     n_points = points_3d.shape[0]

#     n = 9 * n_cameras + 3 * n_points
#     m = 2 * points_2d.shape[0]

#     print("n_cameras: {}".format(n_cameras))
#     print("n_points: {}".format(n_points))
#     print("Total number of parameters: {}".format(n))
#     print("Total number of residuals: {}".format(m))
#     opt = False
   

#     if (opt):
#         A = bundle_adjustment_sparsity(n_cameras, n_points, camera_indices,
#                                        point_indices)
#         # print(camera_params.ravel)
#         x0 = np.hstack((camera_params.ravel(), points_3d.ravel()))

#         res = least_squares(
#             fun,
#             x0,
#             jac_sparsity=A,
#             verbose=2,
#             x_scale='jac',
#             ftol=1e-4,
#             method='trf',
#             args=(n_cameras, n_points, camera_indices, point_indices,
#                   points_2d))

#         parameters = res.x

#         camera_p = np.reshape(parameters[0:camera_params.size], (n_cameras, 9))

#         X = np.reshape(parameters[camera_params.size:], (n_points, 3))

#         for i in range(n_cameras):
#             Q0[0] = camera_p[i, 0]
#             Q0[1] = camera_p[i, 1]
#             Q0[2] = camera_p[i, 2]
#             C0[0] = camera_p[i, 2]
#             C0[1] = camera_p[i, 2]
#             C0[2] = camera_p[i, 6]
#             r_temp = Rscipy.from_rotvec([Q0[0], Q0[1], Q0[2]])
#             Rset[i] = r_temp.as_dcm()
#             Cset[i] = [C0[0], C0[1], C0[2]]

#     return Rset, Cset, X