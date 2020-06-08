# -*- coding: utf-8 -*-
"""
Created on Wed Apr 22 08:08:27 2020

@author: Handi
"""
# In[1]:

kfold = StratifiedKFold(n_splits=5)
splits = kfold.split(d, d['CLASS'])

# In[2]:

import pandas as pd

mydict = [{'a': 1, 'b': 2, 'c': 3, 'd': 4},
          {'a': 100, 'b': 200, 'c': 300, 'd': 400},
          {'a': 1000, 'b': 2000, 'c': 3000, 'd': 4000 }]
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

