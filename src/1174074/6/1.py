# -*- coding: utf-8 -*-
"""
Created on Sun Apr 12 23:24:18 2020

@author: Rin
"""
from keras.layers import Dense, Activation
from keras.models import Sequential
import numpy as np
model = Sequential([
    Dense(100, input_dim=np.shape(train_input)[1]),
    Activation('relu'),
    Dense(10),
    Activation('softmax'),
    ])