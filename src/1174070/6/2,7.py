# In[3]: membuat seq NN, layer pertama dense dari 100 neurons
model = Sequential([
    Dense(100, input_dim=np.shape(train_input)[1]),
    Activation('relu'),
    Dense(10),
    Activation('softmax'),
    ])