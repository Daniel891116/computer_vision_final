import os
import sys
import numpy as np


def calculate_dist(label, pred):
    assert label.shape[0] == pred.shape[0], 'The number of predicted results should be the same as the number of ground truth.'
    dist = np.sqrt(np.sum((label-pred)**2, axis=1))
    dist = np.mean(dist)
    return dist

def benchmark(gt_path, pred_path):
    label = np.loadtxt(gt_path)
    pred = np.loadtxt(os.path.join(pred_path), delimiter=' ')  #TODO: Enter your filename here#
    score = calculate_dist(label, pred)
    print(f'Mean Error of {gt_path.split("/")[1]}: {score:.5f}')

if __name__ == '__main__':
    benchmark(sys.argv[1], sys.argv[2])