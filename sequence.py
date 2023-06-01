import os
import glob
import pandas as pd

from frame import Frame
from camera import Camera
from tqdm import tqdm
from typing import Dict, List
from utils.pcd_utils import numpy2pcd, savepcd

class Sequence:
    """
    This is the class of sequence images dataset
    """
    def __init__(self, seq_dir: str, camera_dir: str):
        """
        params:
            dir: directory that contains the sequence frame data
        
        """
        self.cameras = self.get_cameras(camera_dir)
        self.frames = self.get_frames(seq_dir)

    def get_cameras(self, camera_dir: str) -> List[Camera]:
        """
        instantiate 4 types of cameras
        """
        camera_type = ['f', 'fl', 'fr', 'b']
        all_camera = dict()
        for type in camera_type:
            all_camera[type] = Camera(type, camera_dir)
        return all_camera
    
    def get_frames(self, seq_dir: str) -> List[Dict[str,Frame]]:
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
        for frame_dir in tqdm(frame_dirs[0:1]):
            frame = Frame(frame_dir, self.cameras)
            if frame.type not in all_frame.keys():
                all_frame[frame.type] = [frame]
            else:
                all_frame[frame.type].append(frame)
        return all_frame


if __name__ == '__main__':
    seq_dir = './ITRI_dataset/seq1'
    camera_dir = './ITRI_dataset/camera_info/lucid_cameras_x00'
    seq = Sequence(seq_dir=seq_dir, camera_dir=camera_dir)
    print(seq.frames['f'][0].camera)
    print(seq.frames['f'][1].camera.config)
