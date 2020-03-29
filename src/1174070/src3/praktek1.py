# -*- coding: utf-8 -*-
"""
Created on Mon Mar 16 22:52:24 2020

@author: Bakti Qilan
"""
# In[]
import pandas as pd

d = {'col1':[1,2,3,4,5],'col2':[3,5,5,4,2],'col3':[2,3,1,1,4]}
df = pd.DataFrame(data=d)
print("nomor 1")
print("Hasil: ")
print(df)
# In[]
import numpy as np

slur = np.arange(1,26).reshape(5,5)
print("nomor 2")
print(slur)

# In[]
import matplotlib.pyplot as plt

t = np.arange(0., 5., 0.2)
print("nomor 3")
plt.plot(t, t, 'r--', t, t**2, 'bo', t, t**3, 'g^')
plt.show()