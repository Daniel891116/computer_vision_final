import os
from typing import Dict

import cv2
import numpy as np
import yaml
from utils.camera_utils import quaternion_rotation_matrix, euler_from_quaternion

class Camera:
    """
    This is the camera class
    """
    def __init__(self, type: str, dir: str):
        """
        params
            type [b|f|fl|fr]: specify the location of camera
            dir: the directory where contains the camera configuration\n
            e.g.   dir\n
                    ├── camera_extrinsic.yaml\n
                    ├── gige_100_b_hdr_camera_info.yaml\n
                    ├── gige_100_b_hdr_mask.png\n
                    ├── gige_100_f_hdr_camera_info.yaml\n
                    ├── gige_100_f_hdr_mask.png\n
                    ├── gige_100_fl_hdr_camera_info.yaml\n
                    ├── gige_100_fl_hdr_mask.png\n
                    ├── gige_100_fr_hdr_camera_info.yaml\n
                    └── gige_100_fr_hdr_mask.png\n

        """
        self.type = type
        self.config = self.get_config(type, dir)
        self.mask = self.get_mask(type, dir)
        self.transform = self.get_transform(type, dir)
        
    def get_config(self, type: str, dir: str) -> Dict:
        config_file = os.path.join(dir, f"gige_100_{type}_hdr_camera_info.yaml")
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
        _config = dict()
        _config["camera_matrix"] = self.dict2arr(config["camera_matrix"])
        _config["distortion_coefficients"] = self.dict2arr(config["distortion_coefficients"])
        _config["rectification_matrix"] = self.dict2arr(config["rectification_matrix"])
        _config["projection_matrix"] = self.dict2arr(config["projection_matrix"])
        return _config
        
    def get_mask(self, type: str, dir: str) -> np.ndarray:
        mask_file = os.path.join(dir, f"gige_100_{type}_hdr_mask.png")
        return cv2.imread(mask_file)

    def dict2arr(self, dict: Dict) -> np.array:
        _arr = np.array(dict["data"])
        return np.reshape(_arr, (dict["rows"], dict["cols"]))
    
    def get_transform(self, type: str, dir: str) -> np.array:
        """
        get the transform matrix from itself to base link
        """
        config_file = os.path.join(dir, "camera_extrinsic.yaml")
        transform = None
        with open(config_file, 'r') as f:
            transforms = yaml.safe_load(f)
        for key in transforms.keys():
            if type == key.split('_')[0]:
                transform = np.array(transforms[key]["args"])
                break
        assert transform is not None, "cannot find corresponding extrinsic matrix of the camera"
        extrinsic_matrix = np.zeros((4, 4), dtype=np.float32)
        extrinsic_matrix[0:3, 0:3] = quaternion_rotation_matrix(transform[3:]) # return a 3x3 rotation matrix
        extrinsic_matrix[0:3, 3] = transform[:3]
        extrinsic_matrix[3, 3] = 1
        return extrinsic_matrix
    
if __name__ == '__main__':
    camera = Camera('f', "./ITRI_dataset/camera_info/lucid_cameras_x00")
    print(camera.transform)