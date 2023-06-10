import os
from typing import Dict, List, Tuple, Union

import cv2
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image
from segment import SegmentAnythingWorker
from segment_cv2 import BaseSegmentWorke
from utils.feature_utils import (filter_keypoints, get_SIFT_descriptor,
                                 get_SIFT_features)
from utils.marker_utils import read_road_marker

class Frame():
    """
    This is the class of single frame
    """
    def __init__(self, dir: str, cameras: Dict, segmentor: Union[BaseSegmentWorke, SegmentAnythingWorker]):
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
        # self.timestamp = float(dir.split('/')[-2].replace('_', '.'))
        self.timestamp = dir.split('/')[-2]
        self.image = Image.open(os.path.join(dir, 'raw_image.jpg'))
        if isinstance(segmentor, BaseSegmentWorke):
            detection_dict = self.prepare_detection_data()
            self.contours, _ = segmentor(self.image, detection_dict)
        else:
            contour_file = os.path.join(self.dir, "contours.json")
            if os.path.exists(contour_file):
                with open(contour_file, 'r') as f:
                    try:
                        self.contours = json.load(f)
                    except:
                        detection_dict = read_road_marker(os.path.join(self.dir, "detect_road_marker.csv"))
                        self.contours = segmentor(self.image, detection_dict)
                        for key in self.contours.keys():
                            for i in range(len(self.contours[key])):
                                self.contours[key][i] = self.contours[key][i].tolist()
                        with open(contour_file, 'w') as f:
                            json.dump(self.contours, f, indent=4)
            else:
                detection_dict = read_road_marker(os.path.join(self.dir, "detect_road_marker.csv"))
                self.contours = segmentor(self.image, detection_dict)
                for key in self.contours.keys():
                    for i in range(len(self.contours[key])):
                        self.contours[key][i] = self.contours[key][i].tolist()
                with open(contour_file, 'w') as f:
                    json.dump(self.contours, f, indent=4)
        
        self.image = np.asarray(self.image)
        # self.keypoints, self.descriptors = self.get_filtered_keypoints(visualize=True)
        (self.keypoints, self.descriptors) = get_SIFT_descriptor(self.image, self.contours)
        self.matches = None
        

    def prepare_detection_data(self) -> Dict:
        detect_label_num = 3
        try:
            detection_data = pd.read_csv(os.path.join(self.dir, 'detect_road_marker.csv'), header=None).values
        except pd.errors.EmptyDataError:
            print(self.dir)
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
