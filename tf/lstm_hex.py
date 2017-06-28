import os
import time

import keras
import numpy as np
import tensorflow as tf
from keras.layers import Dense, LSTM, Dropout
from keras.models import Sequential
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score

from deferred import dataload_hex
from deferred.loader2 import loaderTrain, loaderTest

NB_NOTES_READ = dataload_hex.MIN_SIZE

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
print('TensorFlow version: {0}'.format(tf.__version__))

t = time.time()

X_train, X_test, y_train, y_test = loaderTrain.songs[:], loaderTest.songs[:], loaderTrain.artists[
                                                                              :], loaderTest.artists[:]


def toArray(X,y):
    inds = []
    for song in X:
        if len(song) >= NB_NOTES_READ:
            inds.append(X.index(song))
    return np.array([[msg for msg in X[k][:NB_NOTES_READ]] for k in inds], dtype=np.float32), \
           np.array(y)[inds]

X_train, y_train = toArray(X_train, y_train)
X_test, y_test = toArray(X_test, y_test)

# y_test = extend_y(y_test)    # not in keras
# y_train = extend_y(y_train)
# y_train, y_test = np.array(y_train),np.array(y_test)

# learning parameters
learning_rate = 0.001
epoches = 20
batch_size = 200
display_step = 10

# network parameters
n_input = 3  # was 1, changed it to get 3 information on each note
n_steps = NB_NOTES_READ  # was 10
n_hidden = 500
n_classes = 2

# Keras implementation
# Model definition
# 1st model : LSTM

# GBM
clf = GradientBoostingClassifier()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("GBM accuracy = ", accuracy, "Precision = ", precision_score(y_test, y_pred))

model = Sequential([
    LSTM(128, return_sequences=True,input_shape=(NB_NOTES_READ, 4,)),
    Dropout(0.25),
    LSTM(128,return_sequences=False),
    Dropout(0.25),
    Dense(128,activation="relu"),
    Dense(1, activation="sigmoid")
])

# 2nd model : Conv1D
'''model = Sequential([
    Conv1D(32, 40, activation="relu", strides=4 ,input_shape=(NB_NOTES_READ,4)),
    Conv1D(32, 40, activation="relu", strides=4),
    MaxPooling1D(pool_size=2, strides=2, padding="valid"),
    Conv1D(64, 10, activation="relu", strides=2),
    MaxPooling1D(pool_size=2, padding="valid", strides=2),
    Conv1D(128, 3, activation="relu"),
    GlobalAveragePooling1D(),
    Dense(1024,activation="relu"),
    Dropout(0.5),
    Dense(1024,activation="relu"),
    Dropout(0.5),
    Dense(1,activation="sigmoid")
])'''

print("X_train.shape : {0}\nX_test.shape : {1}".format(X_train.shape, X_test.shape))
print("y_train.shape : {0}\ny_test.shape : {1}".format(y_train.shape, y_test.shape))

# Dense (aka Fully connected) 3 times in a row
'''for k in range(3):
    model.add(Dense(256,activation='relu'))'''

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

# Saving the model
model.save("lstm_hex_trained_jazz_rand3.h5")
