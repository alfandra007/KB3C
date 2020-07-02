# -*- coding: utf-8 -*-
"""
Created on Fri May 15 13:47:30 2020

@author: Aulyardha Anindita
"""

# In[]
import scipy.io as io
voxels = io.loadmat("C:/Users/USER/Downloads/3DShapeNetsCode/3DShapeNets/volumetric_data/sofa/30/test/chair_000000008_1.mat")['instance']
# In[]
import numpy as np
voxels = np.pad(voxels, (1, 1), 'constant', constant_values=(0, 0))
# In[]
import scipy.ndimage as nd
voxels = nd.zoom(voxels, (2, 2, 2), mode='constant', order=0)
# In[]
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D 
fig = plt.figure()
ax = Axes3D(fig)
ax.voxels(voxels, edgecolor="black")

plt.show()
plt.savefig('data')
