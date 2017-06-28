import os

import keras
import numpy as np
from keras.layers import Dense, LSTM, Dropout
from keras.models import Sequential

from deferred import dataload_hex
from deferred.loader import loaderTrain, loaderTest


def lstm(batch_size, n_hidden_lstm1, n_hidden_lstm2, n_dropout1, n_dropout2, n_dense, epoches=20):
    """

    :param epoches: 5 < x < 50 but we chose 20
    :param batch_size: 20 < x < 400
    :param n_hidden_lstm1: 20 < x < 512
    :param n_hidden_lstm2: 20 < x < 512
    :param n_dropout1: 0.1 < x < 0.9
    :param n_dropout2: 0.1 < x < 0.9
    :param n_dense: 20 < x < 1024
    :return: string, accuracy between 0 and 100
    """

    NB_NOTES_READ = dataload_hex.MIN_SIZE

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    X_train, X_test, y_train, y_test = loaderTrain.songs[:], loaderTest.songs[:], loaderTrain.artists[
                                                                                  :], loaderTest.artists[:]

    def toArray(X, y):
        inds = []
        for song in X:
            if len(song) < NB_NOTES_READ:
                del (y[X.index(song)])
                del (X[X.index(song)])
            else:
                inds.append(X.index(song))
        return np.array([[msg for msg in song[:NB_NOTES_READ]] for song in X], dtype=np.float32)[inds], \
               np.array(y)[inds]

    X_train, y_train = toArray(X_train, y_train)
    X_test, y_test = toArray(X_test, y_test)

    # Keras implementation
    # Model definition

    model = Sequential([
        LSTM(n_hidden_lstm1, return_sequences=True, input_shape=(NB_NOTES_READ, 4,)),
        Dropout(n_dropout1),
        LSTM(n_hidden_lstm2, return_sequences=False),
        Dropout(n_dropout2),
        Dense(n_dense, activation="relu"),
    ])

    # Output (parameter is 1 because it is a classification problem
    model.add(Dense(1, activation="sigmoid", name="main_output"))

    # Custom optimizer
    sgd = keras.optimizers.SGD(lr=0.0001)

    # Choice of optimizer, loss and metrics
    model.compile(optimizer=sgd,
                  loss='binary_crossentropy',
                  metrics=['binary_accuracy'])

    # Training
    history = model.fit(X_train, y_train, epochs=epoches, batch_size=batch_size, verbose=0)   # no log

    # Testing
    # score = model.evaluate(X_test, y_test, batch_size=32)

    accuracy = history.history.get('binary_accuracy')[-1]

    needed = int(round(accuracy*100, 0))
    return str(needed)