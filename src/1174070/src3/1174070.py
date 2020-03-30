# -*- coding: utf-8 -*-
"""
Created on Fri Mar 13 16:40:43 2020

@author: Bakti Qilan
"""
# In[]
import pandas as pd

dataset = pd.read_csv('E:/backup/sem 6/Kecerdasan Buatan/KB3C - Copy/src/1174083/src3/case.csv', sep=',')
dataset.head()

# In[]
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

y_actu = pd.Series([2, 0, 2, 2, 0, 1, 1, 2, 2, 0, 1, 2], name='Actual')
y_pred = pd.Series([0, 0, 2, 1, 0, 2, 1, 0, 2, 0, 2, 2], name='Predicted')
df_confusion = pd.crosstab(y_actu, y_pred, rownames=['Actual'], colnames=['Predicted'], margins=True)

def plot_confusion_matrix(df_confusion, title='Confusion matrix', cmap=plt.cm.gray_r):
    plt.matshow(df_confusion, cmap=cmap) # imshow
    #plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(df_confusion.columns))
    plt.xticks(tick_marks, df_confusion.columns, rotation=45)
    plt.yticks(tick_marks, df_confusion.index)
    plt.ylabel(df_confusion.index.name)
    plt.xlabel(df_confusion.columns.name)

plot_confusion_matrix(df_confusion)
# In[]

import pandas as pd
df = pd.read_csv("E:/backup/sem 6/Kecerdasan Buatan/KB3C - Copy/src/1174083/src3/movies_metadata.csv", error_bad_lines=False)
#Set the index to the title
df = df.set_index('title')
#Details of the movie 'Grumpier Old Men'
result = df.loc['Grumpier Old Men']
print("Details of the movie 'Grumpier Old Men:")
print(result)