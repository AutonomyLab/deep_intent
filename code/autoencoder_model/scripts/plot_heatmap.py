from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import numpy.random
import matplotlib.pyplot as plt
import ast
from config_aa import *

data = np.load(os.path.join(TEST_DATA_DIR, 'attention_weights.npy'))
print (data.shape)
# Generate some test data
frame = data[0]
x = frame[0]
y = frame[1]

heatmap, xedges, yedges = np.histogram2d(x, y, bins=50)
extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]

plt.clf()
plt.imshow(heatmap.T, extent=extent, origin='lower')
plt.show()