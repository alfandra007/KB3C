# -*- coding: utf-8 -*-
"""
Created on Mon May 18 08:15:28 2020

@author: rezas
"""

import scipy.io as io
import matplotlib.pyplot as plt
import scipy.ndimage as nd
import numpy as np
from mpl_toolkits.mplot3d import Axes3D 

voxels = io.loadmat("D:/New folder/KB3C/src/1174084/8/3DShapeNets/volumetric_data/bed/30/test/bed_000000000_1.mat")['instance']
voxels = np.pad(voxels, (1, 1), 'constant', constant_values=(0, 0))
voxels = nd.zoom(voxels, (2, 2, 2), mode='constant', order=0)

fig = plt.figure()
ax = Axes3D(fig)
ax.voxels(voxels, edgecolor="chartreuse")

plt.show()
plt.savefig('bed')
