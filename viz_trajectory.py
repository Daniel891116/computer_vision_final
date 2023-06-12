import matplotlib.pyplot as plt
import os
import numpy as np
from scipy.ndimage import gaussian_filter1d
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_file", type=str,
        help="your pred_pose.txt")
    parser.add_argument("--smoothe", type=bool,
        help="set this flag to smoothe the output trajectory")
    
    args = parser.parse_args()
    data = np.loadtxt(args.pred_file, delimiter = ' ')
    smoothed_x = gaussian_filter1d(data[:, 0], sigma=8)
    smoothed_y = gaussian_filter1d(data[:, 1], sigma=8)
    c = range(data.shape[0])
    plt.title(os.path.basename(args.pred_file))
    plt.scatter(data[:, 0], data[:, 1], s=1, c = c, alpha=1, cmap = 'Reds')
    if args.smoothed:
        plt.scatter(smoothed_x, smoothed_y, s=1, c = c, alpha=1, cmap = 'Blues')
    plt.savefig(f"{args.pred_file.split('.')[0]}.png")
    plt.show()

if __name__ == '__main__':
    main()