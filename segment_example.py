import cv2
import numpy as np
from segment_cv2 import BaseSegmentWorke
import os
import pandas as pd
from PIL import Image
import torch
import argparse
from distutils.util import strtobool

parser = argparse.ArgumentParser()
parser.add_argument("--seq_path", type=str,
    help="seq path, some thing like /your/path/ITRI_dataset/seq1")
parser.add_argument("--eps_coef", type=float, default=0.02,
        help="approx eps coef, small->detailed, big->rough")
parser.add_argument("--contour_thres", type=float, default=0.8,
        help="contour threshold (number of std), small->light threshold, big->strict threshold")
parser.add_argument("--detect_label_num", type=int, default=1,
    help="number of labels that are detected. 2 means the first two categories, which is zebracross and stopline")
parser.add_argument("--show_contours", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="True to show the contours image. When showing, press anything to continue, press esc successively to terminate the showing")

args = parser.parse_args()

segmentor = BaseSegmentWorke(eps_coef=args.eps_coef, thres=args.contour_thres, show_contour=args.show_contours)
root_path = os.path.join(args.seq_path, 'dataset')
names = os.listdir(root_path)
names.sort()

for route in names:
    detection_route = os.path.join(root_path, route, 'detect_road_marker.csv')
    image_route = os.path.join(root_path, route, 'raw_image.jpg')
    try:
        image = Image.open(image_route)
        # image = cv2.imread(image_route)
        detection_file = pd.read_csv(detection_route)
        detection_data = detection_file.values
        detection_data = np.stack([detection_data[i] for i in range(detection_data.shape[0]) if detection_data[i, 4]<args.detect_label_num])
        # print(detection_data.shape)
        detection_dict = {'boxes': torch.from_numpy(detection_data[:, :4]), 'labels': torch.from_numpy(detection_data[:, 4])}
    except:
        continue
    
    #############################################################
    # contours: list contour points (ndarray)                   #
    # contour points: ndarray, (num_points, 1, 2)               #
    #############################################################
    contours, terminated = segmentor(image, detection_dict)
    
    if terminated: break