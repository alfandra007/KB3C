# In[44]: Soal1

import pandas as gani 
#melakukan import pada library pandas sebagai gani

marvel = {"Marvel" : ['Black Panther','Captain Marvel','Spiderman','Thor']} 
#membuat varibel yang bernama marvel, dan mengisi dataframe nama2 karakter marvel
x = gani.DataFrame(marvel) 
#variabel x membuat DataFrame dari library pandas dan akan memanggil variabel laptop. 
print (' Marvel kesukaan gani ' + x) 
#print hasil dari x

# In[44]: Soal2

import numpy as gani
#melakukan import numpy sebagai gani

matrix_x = gani.eye(10) 
#untuk membuat matrix dengan numpy dengan menggunakan fungsi eye
matrix_x 
#untuk deklrasikan matrix_x yang telah dibuat

print (matrix_x) 
#menampilkan print matrix_x yang telah dibuat dengan 10x10


# In[44]: Soal3

import matplotlib.pyplot as gani
#import matploblib sebagai gani

gani.plot([1,1,7,4,0,7,1]) 
#untuk memberikan nilai plot atau grafik pada gani
gani.xlabel('Muhammad Abdul Gani Wijaya') 
#untuk memberikan label pada x
gani.ylabel('1174071') 
#untuk memberikan label pada y
gani.show() 
#menampilkan print hasil plot berbentuk grafik