# -*- coding: utf-8 -*-
"""
Created on Sun Mar 22 21:07:44 2020

@author: rezas
"""

# In[1]:
import pandas as pd #mengimport librari padas 
us = pd.read_csv("D:/Git Kecerdasan Buatan/KB3C/src/1174084/4/us-500.csv") # variabel us untuk membaca file csv menggunakan fungsi read csv dari padas
print(len(us)) #melihat jumlah dari baris data yang telah di import
print(us.head()) #melihat lima baris pertama data yang telah di import 
print(us.shape) #mengetahui banyak baris dan kolom dari data

# In[5]: 
data_training = us[:450] #membuat data training sebanyak 450 baris
data_testing = us[450:] #membuat data testing dari hasil pengurangan 500-450