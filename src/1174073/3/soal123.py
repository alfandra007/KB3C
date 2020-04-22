# In[44]: Soal1

import pandas as ainul #melakukan import pada library pandas sebagai ainul

laptop = {"Nama Laptop" : ['Asus','HP','Lenovo','Samsung']} #membuat varibel yang bernama laptop, dan mengisi dataframe nama2 laptop
x = ainul.DataFrame(laptop) #variabel x membuat DataFrame dari library pandas dan akan memanggil variabel laptop. 
print (' ainul Punya Laptop ' + x) #print hasil dari x

# In[44]: Soal2

import numpy as ainul #melakukan import numpy sebagai ainul

matrix_x = ainul.eye(10) #membuat matrix dengan numpy dengan menggunakan fungsi eye
matrix_x #deklrasikan matrix_x yang telah dibuat

print (matrix_x) #print matrix_x yang telah dibuat dengan 10x10


# In[44]: Soal3

import matplotlib.pyplot as ainul #import matploblib sebagai ainul

ainul.plot([1,1,7,4,0,7,3]) #memberikan nilai plot atau grafik pada ainul
ainul.xlabel('ainul filiani') #memberikan label pada x
ainul.ylabel('1174073') #memberikan label pada y
ainul.show() #print hasil plot berbentuk grafik