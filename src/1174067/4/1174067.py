# -*- coding: utf-8 -*-
"""
Created on Sun Mar 21 01:31:58 2020

@author: Kaka Kamaludin
"""
# In[]
import pandas as pd #import package pandas, lalu dialiaskan menjadi pd.
csv = pd.read_csv('D:/OneDrive - Hybi.god/KULIAH/Semester 6/AI/KB3C/src/1174067/4/csv.csv', sep=',') #membaca file csv dimana data pada file csv dipisahkan oleh koma, lalu ditampung di variable csv.
# In[]
csv1, csv2 = csv[:450], csv[450:] #membagi data menjadi dua bagian, variable csv1 untuk menampung 450 baris data pertama, variable csv2 untuk menampung 50 baris data terakhir.