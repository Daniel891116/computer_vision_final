import matplotlib.pyplot as plt
import sys
import os
import numpy as np
from scipy.ndimage import gaussian_filter1d


pred_file = sys.argv[1]
data = np.loadtxt(pred_file, delimiter = ' ')
smoothed_x = gaussian_filter1d(data[:, 0], sigma=8)
smoothed_y = gaussian_filter1d(data[:, 1], sigma=8)
c = range(data.shape[0])
plt.title(os.path.basename(pred_file))
plt.scatter(data[:, 0], data[:, 1], s=1, c = c, alpha=1, cmap = 'Reds')
plt.scatter(smoothed_x, smoothed_y, s=1, c = c, alpha=1, cmap = 'Blues')
plt.savefig(f"{pred_file.split('.')[0]}.png")
plt.show()
