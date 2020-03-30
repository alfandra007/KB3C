# -*- coding: utf-8 -*-
"""
Created on Sun Mar 22 11:31:58 2020

@author: Bakti Qilan
"""
# In[]
import pandas as pd #import package pandas, lalu dialiaskan menjadi pd.
marvel = pd.read_csv('E:/backup/sem 6/Kecerdasan Buatan/KB3C - Copy/src/1174083/src4/Marvel.csv', sep=',') #membaca file csv dimana data pada file csv dipisahkan oleh koma, lalu ditampung di variable marvel.
# In[]
marvel1, marvel2 = marvel[:450], marvel[450:] #membagi data menjadi dua bagian, variable marvel1 untuk menampung 450 baris data pertama, variable marvel2 untuk menampung 50 baris data terakhir.