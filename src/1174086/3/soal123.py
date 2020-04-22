# -*- coding: utf-8 -*-
"""
Created on Mon Mar 23 15:03:30 2020

@author: Tia
"""

# In[44]: Soal1

import pandas as tia #melakukan import pada library pandas sebagai tia

makanan = {"Makanan" : ['Pizza','Batagor','Cimol','Lumpia']} #membuat varibel yang bernama boyband, dan mengisi dataframe nama2 makanan
x = tia.DataFrame(makanan) #variabel x membuat DataFrame dari library pandas dan akan memanggil variabel laptop. 
print (' Makanan kesukaan tia' + x) #print hasil dari x

# In[44]: Soal2

import numpy as tia #melakukan import numpy sebagai tia

matrix_x = tia.eye(10) #membuat matrix dengan numpy dengan menggunakan fungsi eye
matrix_x #deklrasikan matrix_x yang telah dibuat

print (matrix_x) #print matrix_x yang telah dibuat dengan 10x10


# In[44]: Soal3

import matplotlib.pyplot as tia #import matploblib sebagai tia

tia.plot([1,1,7,4,0,8,6]) #memberikan nilai plot atau grafik pada tia
tia.xlabel('Tia Nur Candida') #memberikan label pada x
tia.ylabel('1174086') #memberikan label pada y
tia.show() #print hasil plot berbentuk grafik