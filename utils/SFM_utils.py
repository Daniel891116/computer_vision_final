import collections
import glob
import math
import os
import sys
from typing import List, Tuple

import config
import cv2
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
from frame import Frame
from mayavi import mlab
from mpl_toolkits.mplot3d import Axes3D
from scipy.linalg import lstsq
from scipy.optimize import least_squares
from tqdm import tqdm


def match_features(query: np.array, target: np.array, MRT: float=0.7) -> np.ndarray:
    """
    Match two descriptor sets by KNN matching algorithm and Apply Lowe's SIFT matching ratio test (MRT).
    
    param:
        query: descriptors of query keypoints
        target: descriptors of target keypoints
    return:
        matched descriptors
    """
    bf = cv2.BFMatcher(cv2.NORM_L2)
    knn_matches = bf.knnMatch(query, target, k=2)
    matches = []
    for m, n in knn_matches:
        if m.distance < MRT * n.distance:
            matches.append(m)
    return np.array(matches)

def match_all_features(frames: List[Frame]) -> List[cv2.DMatch]:
    """
    match the features of the frame in input list to its adjacent frame.

    param:
        frames: List of sequential frames
    return:
        list of matched features
        %% Notice: the length of the return list would be the length of input frames - 1
    """
    # matches_for_all = []
    for i in range(len(frames) - 1):
        matches = match_features(frames[i].descriptors, frames[i + 1].descriptors)
        frames[i].matches = matches
        # matches_for_all.append(matches)
    # return matches_for_all
        
######################
#寻找图与图之间的对应相机旋转角度以及相机平移
######################
def find_transform(K, p1, p2) -> Tuple[np.ndarray, np.ndarray, np.ndarray]: 
    """
    find the rotation and transpose corresponding points in two images.

    param:
        K: camera intrinsic matric
        p1: matched feature points of the last image
        p2: matched feature points of the added image
    return:
        R: relative rotation matrix of the corresponding camera pose
        T: relative transpose of the corresponding camera pose
        mask: the inlier mask of the input feature points

    """ 
    
    focal_length = 0.5 * (K[0, 0] + K[1, 1])
    principle_point = (K[0, 2], K[1, 2])
    E,mask = cv2.findEssentialMat(p1, p2, focal_length, principle_point, cv2.RANSAC, 0.999, 1.0)
    cameraMatrix = np.array([[focal_length, 0, principle_point[0]], [0, focal_length, principle_point[1]], [0, 0, 1]])
    pass_count, R, T, mask = cv2.recoverPose(E, p1, p2, cameraMatrix, mask)
    
    return R, T, mask

def get_matched_points(p1: cv2.KeyPoint, p2: cv2.KeyPoint, matches: List) -> Tuple[np.ndarray, np.ndarray]:
    
    src_pts = np.asarray([p1[m.queryIdx].pt for m in matches])
    dst_pts = np.asarray([p2[m.trainIdx].pt for m in matches])

    return src_pts, dst_pts

#选择重合的点
def maskout_points(p1: np.ndarray, mask: List) -> np.ndarray:
    p1_copy = []
    for i in range(len(mask)):
        if mask[i] > 0:
            p1_copy.append(p1[i])
    
    return np.array(p1_copy)
    
def init_structure(K, key_points_for_all, matches_for_all):
    p1, p2 = get_matched_points(key_points_for_all[0], key_points_for_all[1], matches_for_all[0])
    
    if find_transform(K, p1, p2):
        R,T,mask = find_transform(K, p1, p2)
    else:
        R,T,mask = np.array([]), np.array([]), np.array([])

    # p1 = maskout_points(p1, mask)
    # p2 = maskout_points(p2, mask)
    
    R0 = np.eye(3, 3)
    T0 = np.zeros((3, 1))
    structure = reconstruct(K, R0, T0, R, T, p1, p2)
    rotations = [R0, R]
    motions = [T0, T]
    correspond_struct_idx = []
    for key_p in key_points_for_all:
        correspond_struct_idx.append(np.ones(len(key_p)) * -1) 
    matches = matches_for_all[0]
    for i, match in enumerate(matches):
        if mask[i] == 0:
            continue
        correspond_struct_idx[0][int(match.queryIdx)] = i
        correspond_struct_idx[1][int(match.trainIdx)] = i
    return structure, correspond_struct_idx, rotations, motions
    
#############
#三维重建
#############
def reconstruct(K, R1, T1, R2, T2, p1, p2):
    
    proj1 = np.zeros((3, 4))
    proj2 = np.zeros((3, 4))
    proj1[0:3, 0:3] = np.float32(R1)
    proj1[:, 3] = np.float32(T1.T)
    proj2[0:3, 0:3] = np.float32(R2)
    proj2[:, 3] = np.float32(T2.T)
    fk = np.float32(K)
    proj1 = np.dot(fk, proj1)
    proj2 = np.dot(fk, proj2)
    s = cv2.triangulatePoints(proj1, proj2, p1.T, p2.T)
    structure = []
    
    for i in range(len(s[0])):
        col = s[:, i]
        col /= col[3]
        structure.append([col[0], col[1], col[2]])
    
    return np.array(structure)

###########################
#将已作出的点云进行融合
###########################
def fusion_structure(matches, struct_indices, next_struct_indices, structure, next_structure):
    
    for i,match in enumerate(matches):  
        query_idx = match.queryIdx
        train_idx = match.trainIdx
        struct_idx = struct_indices[query_idx]  
        if struct_idx >= 0:
            next_struct_indices[train_idx] = struct_idx
            continue
        structure = np.append(structure, [next_structure[i]], axis = 0)
        struct_indices[query_idx] = next_struct_indices[train_idx] = len(structure) - 1
    return struct_indices, next_struct_indices, structure

#制作图像点以及空间点
def get_objpoints_and_imgpoints(matches, struct_indices, structure, key_points):
    object_points = []
    image_points = []
    for match in matches:
        query_idx = match.queryIdx
        train_idx = match.trainIdx
        struct_idx = struct_indices[query_idx]  
        if struct_idx < 0: 
            continue
        object_points.append(structure[int(struct_idx)])
        image_points.append(key_points[train_idx].pt)
    
    return np.array(object_points), np.array(image_points)

########################
#bundle adjustment
########################

# 这部分中，函数get_3dpos是原方法中对某些点的调整，而get_3dpos2是根据笔者的需求进行的修正，即将原本需要修正的点全部删除。
# bundle adjustment请参见https://www.cnblogs.com/zealousness/archive/2018/12/21/10156733.html

def get_3dpos(pos, ob, r, t, K):
    dtype = np.float32
    def F(x):
        p,J = cv2.projectPoints(x.reshape(1, 1, 3), r, t, K, np.array([]))
        p = p.reshape(2)
        e = ob - p
        err = e    
                
        return err
    res = least_squares(F, pos)
    return res.x

def get_3dpos_v1(pos,ob,r,t,K):
    p,J = cv2.projectPoints(pos.reshape(1, 1, 3), r, t, K, np.array([]))
    p = p.reshape(2)
    e = ob - p
    if abs(e[0]) > config.x or abs(e[1]) > config.y:        
        return None
    return pos

def bundle_adjustment(rotations, motions, K, correspond_struct_idx, key_points_for_all, structure):
    
    for i in range(len(rotations)):
        r, _ = cv2.Rodrigues(rotations[i])
        rotations[i] = r
    for i in range(len(correspond_struct_idx)):
        point3d_ids = correspond_struct_idx[i]
        key_points = key_points_for_all[i]
        r = rotations[i]
        t = motions[i]
        for j in range(len(point3d_ids)):
            point3d_id = int(point3d_ids[j])
            if point3d_id < 0:
                continue
            new_point = get_3dpos_v1(structure[point3d_id], key_points[j].pt, r, t, K)
            structure[point3d_id] = new_point
    
    return structure

#######################
#作图
#######################

# 这里有两种方式作图，其中一个是matplotlib做的，但是第二个是基于mayavi做的，效果上看，fig_v1效果更好。fig_v2是mayavi加颜色的效果。

def fig(structure, colors):
    colors /= 255
    for i in range(len(colors)):
        colors[i, :] = colors[i, :][[2, 1, 0]]
    fig = plt.figure()
    fig.suptitle('3d')
    ax = fig.gca(projection = '3d')
    for i in range(len(structure)):
        ax.scatter(structure[i, 0], structure[i, 1], structure[i, 2], color = colors[i, :], s = 5)
    ax.set_xlabel('x axis')
    ax.set_ylabel('y axis')
    ax.set_zlabel('z axis')
    ax.view_init(elev = 135, azim = 90)
    plt.show()

def fig_v1(structure):

    mlab.points3d(structure[:, 0], structure[:, 1], structure[:, 2], mode = 'point', name = 'dinosaur')
    mlab.show()

def fig_v2(structure, colors):
    colors /= 255.0
    for i in range(len(structure)):
        
        mlab.points3d(structure[i][0], structure[i][1], structure[i][2], 
            mode = 'point', name = 'dinosaur', color = (colors[i][0], colors[i][1], colors[i][2]))

    mlab.show()
