# -*- coding: utf-8 -*-
"""
Created on Sun Mar 22 10:11:36 2020

@author: Dini Permata Putri
"""

#In[1]: Soal1
import pandas as pd #digunakan untuk mengimport library pandas
dini = pd.read_csv('D:/Mata Kuliah/Tingkat 3/Semester 6/Kecerdasan Buatan/Chapter 3/drakor.csv',sep=';') #digunakan untuk memanggil file csv dan dipisahkan dengan ;
len(dini) #untuk mengetahui jumlah data
print(dini.juduldrakor) #digunakan untuk mengetahui kolom nama dari variabel dini
# In[2]: Soal2
import numpy as np #digunakan untuk mengimport library numpy
episode = np.sum(dini.jumlahepisode) # menggunakan fungsi sum bawaan dari numpy
print(episode) #menampilkan jumlah episode dari variabel episode
# In[3]: Soal3
from matplotlib import pyplot as plt #digunakan untuk mengimport library marplotplib
Pemrograman = [1,2,3,4,5,6,7] #membuat variabel dengan nama pemrograman
RPL =[7,8,6,11,7,8,9] ##membuat variabel dengan nama RPL
Jarkom = [2,3,4,3,2,7,5] ##membuat variabel dengan nama jarkom
MB =[7,8,7,2,2,4,2] #membuat variabel dengan nama MB
SAP = [8,5,7,8,13,7,15] #membuat variabel dengan nama SAP
slices = [7,2,2,13] #membuat objek irisan dari variabel yang sudah dispesifikasikan
activities = ['Pemrograman','RPL','Jarkom','SAP'] #membuat judul dari masing-masing variabel
cols = ['c','m','r','b'] #mengatur warna atau memberikan warna pada setiap item
plt.pie(slices, #untuk mengatur plot pie
  labels=activities,
  colors=cols,
  startangle=0,
  shadow= True,
  explode=(0.1,0.1,0.1,0.1),
  autopct='%1.1f%%')
plt.title('Pie Plot Dini') #memberikan judul pada pie plot
plt.show() #menampilkan graph