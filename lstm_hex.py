import tensorflow as tf
from tensorflow.contrib import rnn
import numpy as np
import os
import dataload_hex
import time
from loader import loaderTrain, loaderTest
import keras
from keras.models import Sequential,Model
from keras.layers import Dense,Activation,LSTM,Conv1D,Conv2D,Flatten,Embedding,Reshape,Input,ConvLSTM2D, TimeDistributed,Dropout

NB_NOTES_READ = dataload_hex.MIN_SIZE

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
print('TensorFlow version: {0}'.format(tf.__version__))

t = time.time()

X_train, X_test, y_train, y_test = loaderTrain.songs[:], loaderTest.songs[:], loaderTrain.artists[
                                                                              :], loaderTest.artists[:]


def toArray(X,y):
    inds = []
    for song in X:
        if len(song) < NB_NOTES_READ:
            del(y[X.index(song)])
            del(X[X.index(song)])
        else:
            inds.append(X.index(song))
    return np.array([[msg for msg in song[:NB_NOTES_READ]] for song in X], dtype=np.float32)[inds], \
           np.array(y)[inds]

X_train, y_train = toArray(X_train, y_train)
X_test, y_test = toArray(X_test, y_test)

# y_test = extend_y(y_test)    # not in keras
# y_train = extend_y(y_train)
# y_train, y_test = np.array(y_train),np.array(y_test)

# learning parameters
learning_rate = 0.0001
epoches = 10
batch_size = 200
display_step = 10

# network parameters
n_input = 3  # was 1, changed it to get 3 information on each note
n_steps = NB_NOTES_READ  # was 10
n_hidden = 500
n_classes = 2

# Keras implementation
# Model definition

model = Sequential([
    LSTM(128, return_sequences=True,input_shape=(500, 4,)),
    Dropout(0.25),
    LSTM(128,return_sequences=False),
    Dropout(0.25),
    Dense(256,activation="relu"),
    Dropout(0.25)
])

print("X_train.shape : {0}\nX_test.shape : {1}".format(X_train.shape, X_test.shape))
print("y_train.shape : {0}\ny_test.shape : {1}".format(y_train.shape, y_test.shape))

# Dense (aka Fully connected) 3 times in a row
'''for k in range(3):
    model.add(Dense(256,activation='relu'))'''

# Output (parameter is 1 because it is a classification problem
model.add(Dense(1,activation="sigmoid",name="main_output"))

# Custom optimizer
sgd = keras.optimizers.SGD(lr=learning_rate)

# Choice of optimizer, loss and metrics
model.compile(optimizer=sgd,
              loss='binary_crossentropy',
              metrics=['binary_accuracy'])

# Training
model.fit(X_train, y_train, epochs=epoches, batch_size=batch_size)

# Testing
score = model.evaluate(X_test, y_test, batch_size=32)

print("\nScore :", score)
print("\nDuration :", time.time() - t)