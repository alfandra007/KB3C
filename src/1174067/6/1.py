from keras.layers import Dense, Activation
from keras.models import Sequential
import numpy as np
model = Sequential([
    Dense(100, input_dim=np.shape(train_input)[1]),
    Activation('relu'),
    Dense(10),
    Activation('softmax'),
    ])