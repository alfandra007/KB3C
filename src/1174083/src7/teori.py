# -*- coding: utf-8 -*-
"""
Created on Sun Apr 12 17:40:57 2020

@author: Bakti Qilan
"""
# In[1]:
kfold = StratifiedKFold(n_splits=5)
splits = kfold.split(d, d['CLASS'])

# In[2]:
import pandas as pd

mydict = [{'w': 6, 'x': 7, 'y': 8, 'z': 9},
          {'w': 600, 'x': 700, 'y': 800, 'z': 900},
          {'w': 6000, 'x': 7000, 'y': 8000, 'z': 9000 }]
df = pd.DataFrame(mydict)
df 

type(df.iloc[0])
df.iloc[0]

# In[3]:
model = Sequential()
model.add(Dense(512, input_shape=(2000,)))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(2))
model.add(Activation('softmax'))

# In[4]:
model.compile(loss='categorical_crossentropy', optimizer='adamax',
	                  metrics=['accuracy'])

# In[4]:

1164050%3+1 