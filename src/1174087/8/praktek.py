# -*- coding: utf-8 -*-
"""
Created on Sun May 17 16:32:50 2020

@author: PandA23
"""
# In[]
import scipy.io as io
import matplotlib.pyplot as plt
import scipy.ndimage as nd
import numpy as np
from mpl_toolkits.mplot3d import Axes3D 

voxels = io.loadmat("3DShapeNets/volumetric_data/guitar/30/test/guitar_000000003_1.mat")['instance']
voxels = np.pad(voxels, (1, 1), 'constant', constant_values=(0, 0))
voxels = nd.zoom(voxels, (2, 2, 2), mode='constant', order=0)

fig = plt.figure()
ax = Axes3D(fig)
ax.voxels(voxels, edgecolor="chartreuse")

plt.show()
plt.savefig('guitar') 