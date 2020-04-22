# In[3]: fitur ektraksi
loss, acc = model.evaluate(test_input, test_labels, batch_size=32)
# In[3]: fitur ektraksi
print("Done!")
print("Loss: %.4f, accuracy: %.4f" % (loss, acc))