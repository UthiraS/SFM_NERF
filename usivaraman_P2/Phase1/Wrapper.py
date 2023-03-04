#!/usr/bin/evn python

"""
RBE/CS Spring 2023: Classical and Deep Learning Approaches for
Geometric Computer Vision
Project 1: Buildings built in minutes - SfM and NeRF Phase 1


Author(s):
Uthiralakshmi Sivaraman (usivaraman@wpi.edu)
Worcester Polytechnic Institute
"""

# Wrapper Code for Phase 1 SFM


# imports 

import cv2
import numpy as np
import matplotlib.pyplot as plt
import argparse
from mpl_toolkits import mplot3d
from scipy.spatial.transform import Rotation 
import scipy

from EstimateFundamentalMatrix import *
from GetInliersRANSAC import *
from EssentialMatrixFromFundamentalMatrix import *
from ExtractCameraPose import *
from LinearTriangulation import *
from DisambiguateCameraPose import *
from NonlinearTriangulation import *
from PnPRANSAC import *
from NonlinearPnP import *

from Utils import *
from PlotUtils import *





#Reading the set of matches for the five images given and extracting features
def features_from_matches():
    """
    Read data from matching.txt and convert them to features 
    """
    path = 'Data'
    num_images = 5
    num_files = num_images- 1
    index = 0
    match_pair1 =[]
    match_pair2 =[]
    rgb_values =[]
    feature_flag = []
    feature_x = []
    feature_y = []
    for i in range(1,num_files+1):
        print("File :",i+1)
        matching_file = path + '/' +'matching'+ str(i) +'.txt'
        mfile = open(matching_file,"r")
        
        #Reading each row in a given text file
        for line_no, row in enumerate(mfile):
            #Skipping the first line of each file since it contains the number of features 
            if(line_no == 0):
                continue
            # getting each element in row 
            column = row.split(" ")
            # print("Column : ",column)
            # finding the number of elements in each row 
            length_column = len(column)
            x_row = np.zeros((1,num_images))
            y_row = np.zeros((1,num_images))
            num_matches_j = column[0]
            red = column[1]
            green = column[2]
            blue = column[3]
            # given x and y
            u_current = column[4]
            v_current = column[5]
            x_row[0,i-1] = float(u_current)
            y_row[0,i-1] = float(v_current)
            flag_row = np.zeros((1,num_images), dtype = int)
            flag_row[0,i-1]= 1
            if(length_column > 7):
                next_index = 6
                while(next_index<length_column-1):
                    next_id = int(column[next_index])
                    # next x and y
                    u_next = float(column[next_index+1])
                    v_next = float(column[next_index+2])
                    next_index = next_index+3
                    match_pair1.append([i,u_current,v_current])
                    match_pair2.append([next_id,u_next,v_next])
                    rgb_values.append([i,red,green,blue])
                    x_row[0, next_id- 1] = u_next
                    y_row[0, next_id - 1] = v_next
                    flag_row[0,next_id-1]= 1
            feature_x.append(x_row)
            feature_y.append(y_row)
            feature_flag.append(flag_row)
    feature_x = np.asarray(feature_x).reshape(-1,num_images)
    feature_y = np.asarray(feature_y).reshape(-1,num_images)
    feature_flag = np.asarray(feature_flag).reshape(-1,num_images)
   
            
    return match_pair1,match_pair2,rgb_values, feature_x,feature_y,feature_flag

            

            
            

                
             


def main():
    # Getting matches from matches.txt 
    match_pair1,match_pair2,rgb_values, feature_x,feature_y,feature_flag = features_from_matches()
    filtered_feature_flag = np.zeros_like(feature_flag) 
    f_matrix = np.empty(shape=(5,5), dtype=object)
    for i in range(0,4): 
        for j in range(i+1,5):

            idx = np.where(feature_flag[:,i] & feature_flag[:,j])
            pts1 = np.hstack((feature_x[idx,i].reshape((-1,1)), feature_y[idx,i].reshape((-1,1))))
            pts2 = np.hstack((feature_x[idx,j].reshape((-1,1)), feature_y[idx,j].reshape((-1,1))))
            idx = np.array(idx).reshape(-1)
            
            if len(idx) > 8:
                F_inliers, inliers_idx = GetInliersRANSAC(pts1,pts2,8000,0.007,idx)                
                f_matrix[i,j] = F_inliers
                filtered_feature_flag[inliers_idx,j] = 1
                filtered_feature_flag[inliers_idx,i] = 1
    
    F12 = f_matrix[0,1]
    print("Fundamental Matrix between first two images: ",F12)
   
    # Getting Camera Calibration K from calibration.txt
    K = np.loadtxt('Data/calibration.txt')
    print("K  :",K)
    # Estimating Essential Matrix from K and F 
    E12 = EssentialMatrixFromFundamentalMatrix(K,F12)

    # Extracting all the Camera Poses from Essential Matrix 
    CSet, RSet = ExtractCameraPose(E12)


    #Using the point correspondences for image 0 and image 1
    idx = np.where(filtered_feature_flag[:,0] & filtered_feature_flag[:,1])
    pts1 = np.hstack((feature_x[idx,0].reshape((-1,1)), feature_y[idx,0].reshape((-1,1))))
    pts2 = np.hstack((feature_x[idx,1].reshape((-1,1)), feature_y[idx,1].reshape((-1,1))))

    # Initialising C1 and R1 as Reference 
    R1_ = np.identity(3)
    C1_ = np.zeros((3,1))


    X = []

    # Finding the world Point X using Linear Triangulation for the 4 camera poses and later disambigutaing 

    for i in range(len(CSet)):
    
        X.append(LinearTriangulation(K,R1_,C1_,RSet[i],CSet[i],pts1,pts2))
    # X = X/X[3]
    X = np.array(X)

    print("X.shape :",X.shape)

    plt.figure("camera disambiguation")
    # plt.set(xlim=(-30, 30), ylim=(-30,30))
    plt.xlim([-40, 40])
    plt.ylim([-30, 30])
    colors = ['red','brown','greenyellow','teal']
    for color, X_c in zip(colors, X):
        plt.scatter(X_c[:,0],X_c[:,2],color=color,marker='.')
    plt.savefig("/home/uthira/Documents/GitHub/SFM_NERF_WORKING/usivaraman_P2/Phase1/Data/IntermediateOutputImages/DisambiguateCameraPose.jpg")
   
    plt.show()
    plt.close()

    # Given World point X and Camera Poses, Disambiguating 
    Rpose,Cpose,Xpose =DisambiguatePose(RSet,CSet,X)
    Xpose = Xpose/Xpose[:,3].reshape(-1,1)
   
    fig = plt.figure("LT, NLT")
    ax = fig.add_subplot()
    plt.xlim([-30, 30])
    plt.ylim([-10, 30])
    plt.scatter(Xpose[:,0], Xpose[:,2],c="b",s=1,label="Linear Triangulation")
    # plt.savefig("/home/uthira/Documents/GitHub/SFM_NERF_WORKING/usivaraman_P2/Phase1/Data/IntermediateOutputImages/Linear_Triangulation.jpg")
    
    # plt.show()
    # plt.close()

    # fig = plt.figure()
    # ax = fig.add_subplot()
    # plt.scatter(Xpose[:,0], Xpose[:,2],c="b",s=1,label="Disambiguate Camera Pose")
    # plt.savefig("/home/uthira/Documents/GitHub/RBE549_SFM_NERF/usivaraman_P2/Phase1/Data/IntermediateOutputImages/DisambiguateCameraPose.jpg")

    # plt.show()
    # plt.close()


    # # Given World Point X and Camera Poses, We refine the world Point using geometric error using Non Linear Triangulation 
    # X_optimized = NonLinearTriangulation(K,pts1, pts2, Xpose,R1_,C1_,Rpose,Cpose)
    X_optimized = NonlinearTriangulation(K,R1_,C1_,Rpose,Cpose,pts1, pts2, Xpose)
    print(X_optimized.shape)
    # X_optimized = X_optimized / X_optimized[:,3].reshape(-1,1)
    # # # #
    # plt.figure("NlT")
    # plt.xlim([-30, 30])
    # plt.ylim([-10, 30])
    plt.scatter(X_optimized[:,0], X_optimized[:,2],c="r",s=1,label="NonLinear_Triangulation")
    # plt.savefig("/home/uthira/Documents/GitHub/SFM_NERF_WORKING/usivaraman_P2/Phase1/Data/IntermediateOutputImages/NonLinear_Triangulation.jpg")

    # plt.show()
    # plt.close()

   
    DrawCameras(R1_,C1_,plt,ax,"1") #Draw 1st camera 
    DrawCameras(Rpose,Cpose,plt,ax,"2")  # Draw 2nd camera
    # plt.xlim([-20, 20])
    # plt.ylim([-15, 15])
    # plt.legend()
    # plt.show()


    total_err1 = []
    for pt1, pt2, X_3d in zip(pts1,pts2,Xpose):
        err1, err2 = ReProjectionError(X_3d,pt1,pt2,R1_,C1_,Rpose,Cpose,K)
        total_err1.append(err1+err2)
    
    mean_err1 = np.mean(total_err1)

    total_err2 = []
    for pt1, pt2, X_3d in zip(pts1,pts2,X_optimized):
        err1, err2 = ReProjectionError(X_3d,pt1,pt2,R1_,C1_,Rpose,Cpose,K)
        total_err2.append(err1+err2)
    
    mean_err2 = np.mean(total_err2)

    print("Between images",0+1,1+1,"Before optimization Linear Triang: ", mean_err1, "After optimization Non-Linear Triang: ", mean_err2)
    
    
    # Creating two variables X_all and X_found 
    # X_all for storing the all world points 
    # X _found for storing values where we have found positive depth has 1 rest has 0 
    # We use this variable for finding points where we have found goos World Points
    X_all = np.zeros((feature_x.shape[0],3))    
    X_found = np.zeros((feature_x.shape[0],1), dtype = int)

    X_all[idx] = Xpose[:,:3]
    X_found[idx] = 1    
    X_found[np.where(X_all[:2]<0)] = 0

    # The first two camera C and R 
    C_set = [C1_,Cpose]
    R_set = [R1_,Rpose]

  

    for i in range(2,5):

        feature_idx_i = np.where(X_found[:,0] & filtered_feature_flag[:,i])
        if len(feature_idx_i[0]) < 8:
            continue

        pts_i = np.hstack((feature_x[feature_idx_i, i].reshape(-1,1), feature_y[feature_idx_i, i].reshape(-1,1)))

        X = X_all[feature_idx_i,:].reshape(-1,3)

        R_init, C_init = PnPRANSAC(X,pts_i,K, 6000,200)
        # print("C_init:",C_init)
        linear_error_pnp = reprojectionErrorPnP(X, pts_i, K, R_init, C_init)

        Ri, Ci = NonLinearPnp(K, pts_i, X, R_init, C_init)
        print("Ri",Ri)
        print("Ci",Ci)
       
        non_linear_error_pnp = reprojectionErrorPnP(X, pts_i, K, Ri, Ci)
        print("Initial linear PnP error: ", linear_error_pnp, " Final Non-linear PnP error: ", non_linear_error_pnp)

        C_set.append(Ci)
        R_set.append(Ri)


        for k in range(0,i):
            idx_X_pts = np.where(filtered_feature_flag[:,k] & filtered_feature_flag[:,i])
            idx_X_pts = np.asarray(idx_X_pts)
            idx_X_pts = np.squeeze(idx_X_pts)
            if (len(idx_X_pts)<8):
                continue

            x1 = np.hstack((feature_x[idx_X_pts,k].reshape(-1,1), feature_y[idx_X_pts,k].reshape(-1,1)))
            x2 = np.hstack((feature_x[idx_X_pts,i].reshape(-1,1), feature_y[idx_X_pts,i].reshape(-1,1)))

          
            X_d = LinearTriangulation(K,R_set[k],C_set[k],Ri,Ci,x1,x2)

            linear_err = []
            pts1 , pts2 = x1, x2
            for pt1, pt2, X_3d in zip(pts1,pts2,X_d):
                err1, err2 = ReProjectionError(X_3d,pt1,pt2,R_set[k],C_set[k],Ri,Ci,K)
                linear_err.append(err1+err2)
        
            mean_linear_err = np.mean(linear_err)
           
            X = NonlinearTriangulation(K,R_set[k],C_set[k],Ri,Ci,x1,x2,X_d)

            X = X/X[:,3].reshape(-1,1)
                
            non_linear_err = []
            for pt1, pt2, X_3d in zip(pts1,pts2,X):
                err1, err2 = ReProjectionError(X_3d,pt1,pt2,R_set[k],C_set[k],Ri,Ci,K)
                non_linear_err.append(err1+err2)

            mean_nonlinear_err = np.mean(non_linear_err)
            print("Linear Triang error: ", mean_linear_err, "Non-linear Triang error: ", mean_nonlinear_err)

            X_all[idx_X_pts] = X[:,:3]
            X_found[idx_X_pts] = 1


    X_found[X_all[:,2]<0] = 0
    print("#############DONE###################")

    feature_idx = np.where(X_found[:,0])
    X = X_all[feature_idx]
    x = X[:,0]
    y = X[:,1]
    z = X[:,2]

    #####2D Plotting
    fig = plt.figure(figsize = (10,10))
    plt.xlim(-4,6)
    plt.ylim(-2,12)
    plt.scatter(x,z,marker='.',linewidths=0.5, color = 'blue')
    for i in range(0, len(C_set)):
        R1 = Rotation.from_matrix(R_set[i]).as_euler("XYZ")
    
        R1 = np.rad2deg(R1)
        plt.plot(C_set[i][0],C_set[i][2], marker=(3,0, int(R1[1])), markersize=15, linestyle='None')

    plt.savefig('/home/uthira/Documents/GitHub/SFM_NERF_WORKING/usivaraman_P2/Phase1/Data/IntermediateOutputImages/'+'2D.png')
    plt.show()


 






    

if __name__ == "__main__":
    main()
