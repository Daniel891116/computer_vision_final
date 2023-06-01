import os 
import cv2
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from segment_cv2 import BaseSegmentWorke
from utils.feature_utils import get_SIFT_features, get_SIFT_descriptor, filter_keypoints
from typing import Dict, List, Tuple

class Frame():
    """
    This is the class of single frame
    """
    def __init__(self, dir: str, cameras: Dict):
        """
        params:
            dir: the directory that contains the information of an single frame
            cameras: dictionary of camera objects
        """
        self.type = None
        self.dir = dir
        with open(os.path.join(dir, 'camera.csv')) as f:
            self.type = f.readlines()[0].split('_')[-2]
        self.camera = cameras[self.type]
        self.timestamp = float(dir.split('/')[-2].replace('_', '.'))
        self.image = Image.open(os.path.join(dir, 'raw_image.jpg'))
        segmentor = BaseSegmentWorke(eps_coef=0.02, thres=0.8, show_contour=True)
        detection_dict = self.prepare_detection_data()
        self.contours, _ = segmentor(self.image, detection_dict)
        self.image = np.asarray(self.image)
        # self.keypoints, self.descriptors = self.get_filtered_keypoints(visualize=True)
        self.keypoints, self.descriptors = get_SIFT_descriptor(self.image, self.contours)
            
    def prepare_detection_data(self) -> Dict:
        detect_label_num = 3
        detection_data = pd.read_csv(os.path.join(self.dir, 'detect_road_marker.csv')).values
        detection_data = np.stack([detection_data[i] for i in range(detection_data.shape[0]) if detection_data[i, 4]<detect_label_num])
        return {'boxes': torch.from_numpy(detection_data[:, :4]), 'labels': torch.from_numpy(detection_data[:, 4])}
    
    # def get_filtered_keypoints(self, visualize) -> Tuple[List, np.ndarray]:
    #     image = np.asarray(self.image)
    #     _kps, _des = get_SIFT_features(image)
    #     if visualize:
    #         fig1 = image.copy()
    #         print(len(_kps))
    #         cv2.drawKeypoints(image.copy(), _kps, fig1, flags=0)
    #     _kps, _des = filter_keypoints(_kps, _des, self.contours)
    #     if visualize:
    #         fig2 = image.copy()
    #         print(len(_kps))
    #         cv2.drawKeypoints(image.copy(), _kps, fig2, flags=0)
    #         fig = np.concatenate([fig1, fig2], axis=1)
    #         plt.imshow(fig)
    #         plt.show()
    #     return (_kps, _des)