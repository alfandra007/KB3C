# -*- coding: utf-8 -*-
"""
Created on Mon Apr 20 01:00:18 2020

@author: Dini Permata Putri
"""
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from sklearn.model_selection import StratifiedKFold
d = ""

kfold = StratifiedKFold(n_splits=5)
splits = kfold.split(d, d['CLASS'])
# In[]:
import pandas as pd

mydict = [{'a': 1, 'b': 2, 'c': 3, 'd': 4},
          {'a': 100, 'b': 200, 'c': 300, 'd': 400},
          {'a': 1000, 'b': 2000, 'c': 3000, 'd': 4000 }]
df = pd.DataFrame(mydict)
df 

type(df.iloc[0])
df.iloc[0]
# In[]:
model = Sequential()
model.add(Dense(512, input_shape=(2000,)))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(2))
model.add(Activation('softmax'))

# In[]:
model.compile(loss='categorical_crossentropy', optimizer='adamax',
	                  metrics=['accuracy'])
