import os
import config
import numpy as np
from tqdm import tqdm
from utils.pcd_utils import numpy2pcd, visualize_pcds, merge_pcd, ICP, csv_reader
from utils.path_utils import check_directory_valid
import argparse
import json
from scipy.interpolate import UnivariateSpline
from scipy.ndimage import gaussian_filter
from scipy.ndimage import gaussian_filter1d
import matplotlib.pyplot as plt
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target_pcd_dir", type=str,
        help="target pointcloud path, eg. CV_pointclouds")
    parser.add_argument("--seq_dir_path", type=str,
        help="output dxdy file, eg. ITRI_dataset/seq1/")
    parser.add_argument("--output_file", type=str,
        help="output dxdy file, eg. CV_submit.csv")
    parser.add_argument("--visualize", type=bool,
        help="whether to visualize the point clouds of each localization timestamp", default=False)
    
    args = parser.parse_args()
    target_pcd_dir = args.target_pcd_dir
    target_timestamps = dict()
    _target_timestamps = dict()
    for type in config.camera_type:
        target_timestamps[type] = [t.split('.')[0] for t in sorted(os.listdir(os.path.join(target_pcd_dir, type)))]
        _target_timestamps[type] = np.array([float(t.replace('_', '.')) for t in target_timestamps[type]])
    
    dxdy = []
    with open(os.path.join("./ITRI_DLC2",args.seq_dir_path, "localization_timestamp.txt"), 'r') as f:
        eval_times = f.readlines()
    all_pcds = []

    for eval_time in tqdm(eval_times):
        eval_time_ = np.array([float(eval_time.replace('_', '.'))])
        sources = []
        for type in config.camera_type:
            delta_times = np.abs(_target_timestamps[type] - eval_time_)
            eval_index = np.argmin(delta_times)
            # Source point cloud
            eval_pcd_dir = os.path.join(target_pcd_dir, type, target_timestamps[type][eval_index])
            # Target point cloud
            target = csv_reader(f"./ITRI_dataset/{args.seq_dir_path}/dataset/{target_timestamps[type][eval_index]}/sub_map.csv")
            target_pcd = numpy2pcd(target)
            # Source point cloud
            #TODO: Read your point cloud here#
            with open(f"{eval_pcd_dir}.json", 'r') as f:
                source = json.load(f)
                source["type"] = type
            sources.append(source)
            
        sources = merge_pcd(sources, 0.7)
        source_pcd = numpy2pcd(sources)
        # Initial pose
        init_pose = csv_reader(f"./ITRI_DLC2/{args.seq_dir_path}/new_init_pose/{eval_time[:-1]}/initial_pose.csv")

        # Implement ICP
        transformation = ICP(source_pcd, target_pcd, threshold=0.7, init_pose=init_pose)
        pred_x = transformation[0,3]
        pred_y = transformation[1,3]
        dxdy.append([pred_x, pred_y])
        #mutiply the transformation matrix of the ICP to the original point cloud data, see if it matches ground truth
        one = np.ones((sources.shape[0],1), dtype=int)
        sources = np.concatenate((sources,one),axis = 1)
        transformed_sources = []
        for row in sources:
            reshaped_row = row.reshape(-1, 1)
            transformed_row = np.matmul(transformation, reshaped_row)
            transformed_sources.append(transformed_row.ravel())
        sources_transformed = np.array(transformed_sources)
        sources_transformed = sources_transformed[:, :3]
        source_pcd = numpy2pcd(sources_transformed)
        if args.visualize:
            all_pcds.append(source_pcd)
    dxdy = np.array(dxdy)
    print(dxdy.shape)
    # sorted_indices = np.argsort(dxdy[:,0])
    # sorted_x = dxdy[sorted_indices,0]
    # sorted_y = dxdy[sorted_indices,1]
    # spl = UnivariateSpline(sorted_x, sorted_y)
    # dxdy[:,1] = spl(sorted_x)
    x = dxdy[:, 0]
    y = dxdy[:, 1]

    # Apply Gaussian filter separately
    smoothed_x = gaussian_filter1d(x, sigma=8)
    smoothed_y = gaussian_filter1d(y, sigma=8)

    # Combine them back
    smoothed_data = np.column_stack((smoothed_x, smoothed_y))


# Your data
# data = np.random.rand(619, 2)  # Just for example, replace with your actual data

# Separate x and y
    x = dxdy[:, 0]
    y = dxdy[:, 1]

    # def causal_gaussian_filter1d(data, sigma):
    #     result = np.zeros_like(data)
    #     kernel_radius = int(3.0 * sigma + 0.5)  # Define the size of the kernel
    #     for i in range(len(data)):
    #         # Create a Gaussian kernel
    #         kernel = np.exp(-(np.arange(kernel_radius, -1, -1) ** 2) / (2 * sigma ** 2))
    #         kernel /= np.sum(kernel)  # Normalize the kernel

    #         # Apply the kernel to the data
    #         for j in range(kernel_radius + 1):
    #             if i - j < 0:
    #                 break
    #             result[i] += kernel[j] * data[i - j]
    #     return result

    # # Apply the causal Gaussian filter
    # causal_smoothed_x = causal_gaussian_filter1d(x, sigma=8)
    # causal_smoothed_y = causal_gaussian_filter1d(y, sigma=8)

    # # Combine them back
    # causal_smoothed_data = np.column_stack((causal_smoothed_x, causal_smoothed_y))

    plt.plot(smoothed_x, smoothed_y)
    plt.savefig("smoothing.png")
    np.savetxt(args.output_file, smoothed_data, delimiter=' ', fmt='%f')
    if args.visualize:
        visualize_pcds(all_pcds, target_pcd)
if __name__ == "__main__":
    main()