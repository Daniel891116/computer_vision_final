import os
from typing import Dict, List

import numpy as np
from tqdm import tqdm

import config
from camera import Camera
from sequence import Sequence
from utils.pcd_utils import numpy2pcd, savepcd
import argparse

class Reconstruct:
    """
    Reconstruct class create a object that reconstruct the 3D point cloud of road marker from a sequence of images based on pinhole model or SFM method.
    """
    def __init__(self, seq_dir: str, camera_dir: str, seg_method: str='CV'):
        """
        params:
            seg_dir: directory that contains the sequence frame 
            camera_dir: directory that contains the camera info file
            seg_method['CV'|'SAM']: method of segment road marker of frame, default = 'CV'
        """
        self.sequence = Sequence(seq_dir, camera_dir, seg_method)
        self.seg_method = seg_method
        self.all_structure = dict()
        self.structures = dict()

    def __call__(self, reconstruct_method='pinhole'):
        if reconstruct_method == 'pinhole':
            self.__reconstruct()
        else:
            raise NotImplementedError
        
    def __reconstruct(self) -> Dict[str, List]:
        print(f"reconstructing...\nmethod:[{self.seg_method}]")
        for type in config.camera_type:
            self.structures[type] = []
            for frame in tqdm(self.sequence.frames[type]):
                data = {
                    # "timestamp": frame.timestamp,
                    # "debug_img": cv2.drawContours(frame.image, frame.contours, -1, (255, 0, 0), 1),
                    "pcd": self.__perspective_project(frame.camera, frame.keypoints, self.sequence.cameras, 25),
                    "timestamp": frame.timestamp
                }
                self.structures[type].append(data)

        for type in config.camera_type:
            if not os.path.exists(f"./{self.seg_method}_pointclouds/{type}"):
                os.makedirs(f"./{self.seg_method}_pointclouds/{type}")
            # plt.imsave(f"{type}_0.png", self.structures[type][0]["debug_img"])
            for data in self.structures[type]:
                np.savetxt(os.path.join(f"{self.seg_method}_pointclouds",type,f"{data['timestamp']}.csv"), data['pcd'], delimiter=',', fmt='%f')
                # savepcd(os.path.join(f"{self.seg_method}_pointclouds",f"{data['timestamp']}.ply"), numpy2pcd(data["pcd"]))

    def __perspective_project(self, camera: Camera, keypoints: List, cameras: List, range) -> np.ndarray:
        """
        project the keypoints of image to z = -CAMERA_HEIGHT with respect to the base_link

        param:
            camera: the camera object that keypoints belongs to
            keypoints: the keypoints to be projected
            base_transform: the transform from the f to base_link
            range: the maximum projected range with respect to the base_link
        """

        CAMERA_HEIGHT = config.CAMERA_HEIGHT
        structure = []
        K = camera.config['projection_matrix'][0:3, 0:3]
        fx = K[0][0]
        fy = K[1][1]
        cx = K[0][2]
        cy = K[1][2]
        

        if camera.type == 'f':
            base_transform = cameras['f'].transform
            transform = np.linalg.inv(base_transform)
        elif camera.type == 'b':
            base_transform = np.linalg.inv(cameras['fl'].transform) @ np.linalg.inv(cameras['f'].transform)
            transform = np.linalg.inv(camera.transform) @ base_transform
        else:
            base_transform = np.linalg.inv(cameras['f'].transform)
            transform = np.linalg.inv(camera.transform) @ base_transform
        # print(f"base_link_to_{camera.type}:\n{transform}")
        # print(f"f_{camera.type}:\n{np.linalg.inv(camera.transform)}")
        R = transform[0:3, 0:3]
        T = transform[0:3,   3]
        a11 = (fx*R[0][0]+cx*R[2][0])
        a12 = (fx*R[0][1]+cx*R[2][1])
        a21 = (fy*R[1][0]+cy*R[2][0])
        a22 = (fy*R[1][1]+cy*R[2][1])
        a31 = R[2][0]
        a32 = R[2][1]
        a33 = -1
        a34 = R[2][2]*CAMERA_HEIGHT-T[2]
        
        for kp in keypoints:
            (u, v) = kp.pt
            a13 = -u
            a14 = fx*(R[0][2]*CAMERA_HEIGHT-T[0])+cx*(R[2][2]*CAMERA_HEIGHT-T[2])
            a23 = -v
            a24 = fy*(R[1][2]*CAMERA_HEIGHT-T[1])+cy*(R[2][2]*CAMERA_HEIGHT-T[2])
            M = np.array(
                [
                    [a11, a12, a13],
                    [a21, a22, a23],
                    [a31, a32, a33]
                ]
            )
            N = np.array([
                [a14, a24, a34]
            ]).T
            solve = np.linalg.solve(M, N)
            if np.linalg.norm(solve, ord=2) > range:
                continue
            point_base_link = [solve[0][0], solve[1][0], -CAMERA_HEIGHT]              

            structure.append(point_base_link)
        
        return np.array(structure)
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq_dir_path", type=str,
        help="seq path, some thing like /your/path/ITRI_dataset/seq1")
    parser.add_argument("--camera_dir_path", type=str,
        help="directory that contains the camera information. eg. ITRI_dataset/camera_info/lucid_cameras_x00")
    parser.add_argument("--segent_method", type=str,
        help="segmentation method applied to extract road marker. eg. [SAM|CV]")

    
    args = parser.parse_args()
    reconstructor = Reconstruct(args.seq_dir_path, args.camera_dir_path, seg_method=args.segent_method)
    reconstructor('pinhole')
    # base_f = np.linalg.inv(reconstructor.sequence.cameras['f'].transform)
    # base_fl = np.linalg.inv(reconstructor.sequence.cameras['fl'].transform) @ base_f
    # base_fr = np.linalg.inv(reconstructor.sequence.cameras['fr'].transform) @ base_f
    # base_b = np.linalg.inv(reconstructor.sequence.cameras['b'].transform) @\
    #          np.linalg.inv(reconstructor.sequence.cameras['fl'].transform) @\
    #          base_f
    # print(f"base_f:\n{base_f}")
    # print(f"base_fl:\n{base_fl}")
    # print(f"base_fr:\n{base_fr}")
    # print(f"base_b:\n{base_b}")

    # basis = np.zeros((300, 4))
    # axis = np.linspace(0, 10, num=100)
    # for i in range(3):
    #     basis[100*i:100*(i+1), i] = axis.T  
    # basis[:, 3] = 1  
    
    # basis_f = (np.linalg.inv(base_f) @ basis.T).T[:, 0:3]
    # basis_fl = (np.linalg.inv(base_fl) @ basis.T).T[:, 0:3]
    # basis_fr = (np.linalg.inv(base_fr) @ basis.T).T[:, 0:3]
    # basis_b = (np.linalg.inv(base_b) @ basis.T).T[:, 0:3]

    # for i, axis in enumerate(['x', 'y', 'z']):
    #     savepcd(f"basis_f_{axis}.ply", numpy2pcd(basis_f[100*i:100*(i+1)]))
    #     savepcd(f"basis_fl_{axis}.ply", numpy2pcd(basis_fl[100*i:100*(i+1)]))
    #     savepcd(f"basis_fr_{axis}.ply", numpy2pcd(basis_fr[100*i:100*(i+1)]))
    #     savepcd(f"basis_b_{axis}.ply", numpy2pcd(basis_b[100*i:100*(i+1)]))

