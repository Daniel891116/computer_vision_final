import os
from typing import Dict, List

import numpy as np
from tqdm import tqdm

import config
from camera import Camera
from sequence import Sequence
from utils.pcd_utils import numpy2pcd, savepcd
from ITRI_DLC.ICP import ICP

def main():
    target_pcd_dir = "CV_pointclouds"
    target_timestamps = os.listdir(target_pcd_dir)
    target_timestamps = [t.split['.'][0] for t in target_timestamps]
    data_timestamps = "ITRI_dataset/seq1/localization_timestamp.txt"
    with open(data_timestamps, 'r') as f:
         times = f.readlines()
    for time in times:
        if time in 

if __name__ == "__main__":
    main()

