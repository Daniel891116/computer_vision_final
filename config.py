import os
import numpy as np

image_dir = 'ITRI_dataset/seq1/dataset'
# image_dir = 'Herz-Jesus-P25/images/'
# camera_file = 'Herz-Jesus-P25/images/K.txt'
MRT = 0.7
#相机内参矩阵,其中，K[0][0]和K[1][1]代表相机焦距，而K[0][2]和K[1][2]
#代表图像的中心像素。
# with open (camera_file) as f:
#     k = [[float(param) for param in line.split(' ')[0:3]] for line in f.readlines()]
# K = np.array(k)
K = np.array([
    [661.949026684, 0.0, 720.264314891],
    [0.0, 662.758817961, 464.188882538],
    [0.0, 0.0, 1.0]
], dtype = np.float32)
print(K)
#选择性删除所选点的范围。
x = 0.5
y = 1