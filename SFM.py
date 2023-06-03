import config
import cv2
import matplotlib.pyplot as plt
import numpy as np
from sequence import Sequence
from tqdm import tqdm
from utils.SFM_utils import (fig_v1, fusion_structure, get_matched_points,
                             get_objpoints_and_imgpoints, init_structure,
                             reconstruct)
from utils.pcd_utils import numpy2pcd, savepcd

class SFM():
    """
    This is the class of Structure From Motion
    """
    def __init__(self, seq_dir: str, camera_dir: str):
        """
        params:
            dir: directory that contains the sequence frame data
        """
        self.sequence = Sequence(seq_dir, camera_dir)
        self.all_structure = None
        self.structures = dict()
        for type in config.camera_type:
            self.structures[type] = []
    
    def create_sctructure(self) -> np.ndarray:
        """
        This function would create the 3D structure of the sequence and store it in self.structures

        return:
            {
            'lr': 
                [
                    structures
                ],
            'l': 
                [
                    structures
                ]
            }  
        """
        for type in config.camera_type:
            camera = self.sequence.cameras[type]
            frames = self.sequence.frames[type]
            key_points_for_all = [frame.keypoints for frame in frames]
            matches_for_all = [frame.matches for frame in frames[:-1]]
            # drop the matches of last frame because it is None
            K = camera.config["camera_matrix"]
            structure, correspond_struct_idx, rotations, motions = init_structure(K, key_points_for_all, matches_for_all)
            self.structures[type].append(structure)
            for i in tqdm(range(1, len(matches_for_all))):
                object_points, image_points = get_objpoints_and_imgpoints(matches_for_all[i], correspond_struct_idx[i], structure, key_points_for_all[i + 1])
                #在python的opencv中solvePnPRansac函数的第一个参数长度需要大于7，否则会报错
                #这里对小于7的点集做一个重复填充操作，即用点集中的第一个点补满7个
                if len(image_points) < 7:
                    while len(image_points) < 7:
                        object_points = np.append(object_points, [object_points[0]], axis = 0)
                        image_points = np.append(image_points, [image_points[0]], axis = 0)
                
                _, r, T, _ = cv2.solvePnPRansac(object_points, image_points, K, np.array([]), iterationsCount=1000)
                R, _ = cv2.Rodrigues(r)
                rotations.append(R)
                motions.append(T)
                p1, p2 = get_matched_points(key_points_for_all[i], key_points_for_all[i + 1], matches_for_all[i])
                next_structure = reconstruct(K, rotations[i], motions[i], R, T, p1, p2)
                self.structures[type].append(next_structure)
                correspond_struct_idx[i], correspond_struct_idx[i + 1], structure = fusion_structure(matches_for_all[i],correspond_struct_idx[i],correspond_struct_idx[i+1],structure,next_structure)
            self.all_structure = structure

    def visualize_structure(self, type: str) -> None:
        assert len(self.structures[type]) != 0, f"There are no structure of type {type}, you should call create_structure() first."
        for structure in self.structures[type]:
            fig_v1(structure)

    def merge_multiview_sctructure(self):
        """
        This function merge the point clouds of different view from the camera extrinsic matrix
        """
        pass

if __name__ == '__main__':
    sfm = SFM("ITRI_dataset/seq1", "ITRI_dataset/camera_info/lucid_cameras_x00")
    sfm.create_sctructure()
    # fig_v1(sfm.all_structure)
    # for i, structure in enumerate(sfm.structures['f']):
    #     savepcd(f"ITRI_{i}.ply", numpy2pcd(structure))
    savepcd("ITRI_.ply", numpy2pcd(sfm.all_structure))
    # sfm.visualize_structure('f')
