import matplotlib.pyplot as plt
import sys
import os
import numpy as np

pred_file = sys.argv[1]
data = np.loadtxt(pred_file, delimiter = ' ')
c = range(data.shape[0])
plt.title(os.path.basename(pred_file))
plt.scatter(data[:, 0], data[:, 1], s=1, c = c, alpha=1, cmap = 'Reds')
plt.savefig(f"{pred_file.split('.')[0]}.png")
plt.show()
