import os
import sys
import numpy as np


def calculate_dist(label, pred):
    assert label.shape[0] == pred.shape[0], 'The number of predicted results should be the same as the number of ground truth.'
    dist = np.sqrt(np.sum((label-pred)**2, axis=1))
    dist = np.mean(dist)
    return dist



def benchmark(dataset_path, sequences, pred_path):
    for seq in sequences[0:1]:
        label = np.loadtxt(os.path.join(dataset_path, seq, 'gt_pose.txt'))
        pred = np.loadtxt(os.path.join(pred_path), delimiter=' ')  #TODO: Enter your filename here#
        score = calculate_dist(label, pred)
        print(f'Mean Error of {seq}: {score:.5f}')

if __name__ == '__main__':
    dataset_path = 'ITRI_DLC2'
    sequences = ['seq1', 'seq2', 'seq3']
    benchmark(dataset_path, sequences, sys.argv[1])