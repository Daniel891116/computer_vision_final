import glob
import os
from typing import Dict, List

import cv2
import pandas as pd
from tqdm import tqdm

import config
from camera import Camera
from frame import Frame
from segment import SegmentAnythingWorker
from segment_cv2 import BaseSegmentWorke
from utils.SFM_utils import match_all_features


class Sequence():
    """
    This is the class of sequence images dataset
    """
    def __init__(self, seq_dir: str, camera_dir: str, seg_method: str):
        """
        params:
            dir: directory that contains the sequence frame data
        """
        if seg_method == 'SAM':
            self.segmentor = SegmentAnythingWorker("checkpoint/sam_vit_h.pth")
        elif seg_method == 'CV':
            self.segmentor = BaseSegmentWorke(eps_coef=0.02, thres=0.8, show_contour=False)
        else:
            raise NotImplementedError
        self.cameras = self.get_cameras(camera_dir)
        self.frames = self.get_frames(seq_dir)
        # self.match()

    def get_cameras(self, camera_dir: str) -> Dict[str, Camera]:
        """
        instantiate 4 types of cameras
        """
        camera_type = ['f', 'fl', 'fr', 'b']
        all_camera = dict()
        for type in camera_type:
            all_camera[type] = Camera(type, camera_dir)
        return all_camera
    
    def get_frames(self, seq_dir: str) -> Dict[str, List[Frame]]:
        """
        get all frames and additional information in the sequence

        param:
            dir: directory that contains the sequence frame data
        return:
            {   
                "l":
                [
                    <class frame>,...
                ],
                "lr":...
            }
        """
        all_frame = dict()
        frame_dirs = sorted(glob.glob(os.path.join(seq_dir, "dataset", "*/")))
        for frame_dir in tqdm(frame_dirs):
            try:
                pd.read_csv(os.path.join(frame_dir, "detect_road_marker.csv"), header=None)
            except pd.errors.EmptyDataError:
                print("empty")
                continue
            frame = Frame(frame_dir, self.cameras, self.segmentor)
            if frame.type not in all_frame.keys():
                all_frame[frame.type] = [frame]
            else:
                all_frame[frame.type].append(frame)
        return all_frame

    def match(self):
        """
        match all consecutive descriptors in the sequence of frames
        """
        for type in config.camera_type:
            match_all_features(self.frames[type])

if __name__ == '__main__':
    seq_dir = './ITRI_dataset/seq1'
    camera_dir = './ITRI_dataset/camera_info/lucid_cameras_x00'
    seq = Sequence(seq_dir=seq_dir, camera_dir=camera_dir)
    seq.frames['fl'][0].matches = 'test'
    print(seq.frames['fl'][0].matches)
    print(seq.frames['fl'][0].camera)
    frame = seq.frames['f'][1]

    img = cv2.drawKeypoints(frame.image, frame.keypoints, None, flags=0)
    # plt.imshow(img)
    # plt.show()
    img = cv2.drawKeypoints(frame.image, frame.keypoints, None, flags=0)
    # plt.imshow(img)
    # plt.show()
