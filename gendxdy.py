import os
import config
import numpy as np
from tqdm import tqdm
from utils.pcd_utils import numpy2pcd
from ITRI_DLC.ICP import ICP, csv_reader
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target_pcd_dir", type=str,
        help="target pointcloud path, eg. CV_pointclouds")
    parser.add_argument("--output_file", type=str,
        help="output dxdy file, eg. CV_submit.csv")

    args = parser.parse_args()
    target_pcd_dir = args.target_pcd_dir
    target_timestamps = dict()
    _target_timestamps = dict()
    for type in config.camera_type:
        target_timestamps[type] = [t.split('.')[0] for t in sorted(os.listdir(os.path.join(target_pcd_dir, type)))]
        _target_timestamps[type] = np.array([float(t.replace('_', '.')) for t in target_timestamps[type]])

    data_timestamps = "ITRI_dataset/seq1/localization_timestamp.txt"
    dxdy = []
    with open(data_timestamps, 'r') as f:
        eval_times = f.readlines()
    for eval_time in tqdm(eval_times):
        eval_time = np.array([float(eval_time.replace('_', '.'))])
        sources = []
        for type in config.camera_type:
            delta_times = np.abs(_target_timestamps[type] - eval_time)
            eval_index = np.argmin(delta_times)
            eval_pcd_path = os.path.join(target_pcd_dir, type, target_timestamps[type][eval_index])
            # Target point cloud
            target = csv_reader(f"ITRI_dataset/seq1/dataset/{target_timestamps[type][eval_index]}/sub_map.csv")
            target_pcd = numpy2pcd(target)
            # Source point cloud
            #TODO: Read your point cloud here#
            source = csv_reader(f"{eval_pcd_path}.csv")
            if source.shape[0] != 0:
                if source.ndim != 2:
                    source = np.expand_dims(source, axis=0)
                sources.append(source)
            
            
        sources = np.concatenate(sources, axis = 0)
        source_pcd = numpy2pcd(sources)

        # Initial pose
        init_pose = csv_reader(f"ITRI_dataset/seq1/dataset/{target_timestamps[type][eval_index]}/initial_pose.csv")

        # Implement ICP
        transformation = ICP(source_pcd, target_pcd, threshold=0.02, init_pose=init_pose)
        pred_x = transformation[0,3]
        pred_y = transformation[1,3]
        dxdy.append([pred_x, pred_y])
        # print(pred_x, pred_y)
    np.savetxt(args.output_file, np.array(dxdy), delimiter=',', fmt='%f')

if __name__ == "__main__":
    main()

