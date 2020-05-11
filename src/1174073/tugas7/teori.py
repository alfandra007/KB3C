kfold = StratifiedKFold(n_splits=5)
splits = kfold.split(d, d['CLASS'])

model = Sequential()
model.add(Dense(512, input_shape=(2000,)))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(2))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adamax',
              metrics=['accuracy'])
