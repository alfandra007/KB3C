# -*- coding: utf-8 -*-
"""
Created on Tue May 12 22:12:54 2020

@author: Bakti Qilan
"""

import scipy.io as io
import matplotlib.pyplot as plt
import scipy.ndimage as nd
import numpy as np
from mpl_toolkits.mplot3d import Axes3D 

voxels = io.loadmat("/content/3DShapeNets/volumetric_data/flower_pot/30/test/flower_pot_000000010_1.mat")['instance']
voxels = np.pad(voxels, (1, 1), 'constant', constant_values=(0, 0))
voxels = nd.zoom(voxels, (2, 2, 2), mode='constant', order=0)

fig = plt.figure()
ax = Axes3D(fig)
ax.voxels(voxels, edgecolor="chartreuse")

plt.show()
plt.savefig('flower_pot')