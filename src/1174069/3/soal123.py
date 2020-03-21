# In[44]: Soal1

import pandas as fanny #melakukan import pada library pandas sebagai fanny

boyband = {"Boyband" : ['EXO','SEVENTEEN','DAY6','IKON']} #membuat varibel yang bernama boyband, dan mengisi dataframe nama2 boyband
x = fanny.DataFrame(boyband) #variabel x membuat DataFrame dari library pandas dan akan memanggil variabel laptop. 
print (' Boyband kesukaan Fanny ' + x) #print hasil dari x

# In[44]: Soal2

import numpy as fanny #melakukan import numpy sebagai fanny

matrix_x = fanny.eye(10) #membuat matrix dengan numpy dengan menggunakan fungsi eye
matrix_x #deklrasikan matrix_x yang telah dibuat

print (matrix_x) #print matrix_x yang telah dibuat dengan 10x10


# In[44]: Soal3

import matplotlib.pyplot as fanny #import matploblib sebagai fanny

fanny.plot([1,1,7,4,0,6,9]) #memberikan nilai plot atau grafik pada fanny
fanny.xlabel('Fanny Shafira Damayanti') #memberikan label pada x
fanny.ylabel('1174069') #memberikan label pada y
fanny.show() #print hasil plot berbentuk grafik