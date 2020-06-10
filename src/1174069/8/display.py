# -*- coding: utf-8 -*-
"""
Created on Sun May 17 14:56:53 2020

@author: FannyShafira
"""

# In[]
            
import scipy.io as io
voxels = io.loadmat("data/3DShapeNets/volumetric_data/chair/30/test/chair_000000000_1.mat")['instance']
#%%
import numpy as np
voxels = np.pad(voxels, (1, 1), 'constant', constant_values=(0, 0))
#%%
import scipy.ndimage as nd
voxels = nd.zoom(voxels, (2, 2, 2), mode='constant', order=0)
#%%
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D 
fig = plt.figure()
ax = Axes3D(fig)
ax.voxels(voxels, edgecolor="red")

plt.show()
plt.savefig('data')