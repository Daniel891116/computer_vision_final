import argparse

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter1d


def visualize_trajectory(pred_file: str, smoothed: bool):
    data = np.loadtxt(pred_file, delimiter = ' ')
    smoothed_x = gaussian_filter1d(data[:, 0], sigma=8)
    smoothed_y = gaussian_filter1d(data[:, 1], sigma=8)
    c = range(data.shape[0])
    
    if smoothed:
        plt.title("smoothed trajectory")
        plt.scatter(smoothed_x, smoothed_y, s=1, c = c, alpha=1, cmap = 'Blues')
    else:
        plt.title("raw trajectory")
        plt.scatter(data[:, 0], data[:, 1], s=1, c = c, alpha=1, cmap = 'Reds')
    # plt.savefig(f"{pred_file.split('.')[0]}.png")
    plt.show()    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_file", type=str,
        help="your pred_pose.txt")
    parser.add_argument("--smooth", action='store_true',
        help="set this flag to smooth the output trajectory")
    
    args = parser.parse_args()
    visualize_trajectory(args.pred_file, args.smooth)