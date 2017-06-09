import tensorflow as tf
from tensorflow.contrib import rnn
import numpy as np
import os
import dataload
import time
from loader import loaderTrain, loaderTest
import keras as keras
from keras.models import Sequential
from keras.layers import Dense,Activation,LSTM,Conv2D,Flatten

# from tensorflow.examples.tutorials.mnist import input_data
# mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)


NB_NOTES_READ = dataload.MIN_SIZE
NB_TRACKS_READ = 3

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
print('TensorFlow version: {0}'.format(tf.__version__))

t = time.time()


def extend_y(y):
    temp = y.tolist()
    for i in range(len(temp)):
        if temp[i] == 1:
            temp[i] = [0, 1]
        else:
            temp[i] = [1, 0]
    y = np.array(temp)
    return y


# dataset = jl.load(r'C:\Users\Theo\Desktop\ClassifierAlbert\datasetSeries.pkl')
# X_train, X_test, y_train, y_test = dataset

X_train, X_test, y_train, y_test = loaderTrain.songs[:], loaderTest.songs[:], loaderTrain.artists[
                                                                              :], loaderTest.artists[:]


def toArray(X, y):
    """Turns lists X and y into arrays, only using relevant informations and a certain number of tracks and notes.
    This function selects the track that starts the latest and put the temporal starting point of gathering the input
    there.
    Params:
        X: cf dataload
        y: cf dataload
    Return:
         X: array of shape (number of songs kept, NB_TRACKS_READ, NB_NOTES_READ, 3)
         y: array of shape (number of songs kept, 1)
    """
    X_train_tmp = []
    indsTrain = []
    for song in X:
        tmp = []
        cmpt = 0
        if len(song.tracks) > 2:
            song.sort_by_tick()
            start_tick = song.tracks[0].notes[0].duration
            for track in song.tracks:
                tmp_track = []
                if track > NB_NOTES_READ and cmpt < 3:
                    for note in track.notes:
                        if note.tick >= start_tick:  # TODO: make sure there are enough notes.
                            tmp_track.append([note.note, note.tick, note.duration])
                    cmpt += 1
                    tmp.append(tmp_track)
            if cmpt == NB_TRACKS_READ:
                X_train_tmp.append(tmp)
                indsTrain.append(X.index(song))
    return np.array([[[k for k in track[:NB_NOTES_READ]] for track in song] for song in X_train_tmp], dtype=np.float32), \
           np.array(y)[indsTrain]

def toArray16(X, y):
    """Turns lists X and y into arrays, using all tracks and NB_NOTES_READ notes.
    This function selects the track that starts the latest and put the temporal starting point of gathering the input
    there.
    Params:
        X: cf dataload
        y: cf dataload
    Return:
         X: array of shape (number of songs kept, 16, NB_NOTES_READ, 3)
         y: array of shape (number of songs kept, 1)
    """
    X_train_tmp = []
    indsTrain = []
    empty = [[-1,-1,-1] for k in range(NB_NOTES_READ)]
    for song in X:
        tmp = []
        if len(song.tracks) > NB_TRACKS_READ-1:
            song.sort_by_tick()
            start_tick = song.tracks[0].notes[0].duration
            for k in range(16):
                try:
                    track = song.tracks[k]
                    tmp_track = []
                    if track > NB_NOTES_READ:
                        for note in track.notes:
                            if note.tick >= start_tick:  # TODO: make sure there are enough notes.
                                tmp_track.append([note.note, note.tick, note.duration])
                        tmp.append(tmp_track)
                    else:
                        tmp.append(empty)
                except:
                    tmp.append(empty)
            X_train_tmp.append(tmp)
            indsTrain.append(X.index(song))
    return np.array([[[k for k in track[:NB_NOTES_READ]] for track in song] for song in X_train_tmp], dtype=np.float32), \
           np.array(y)[indsTrain]


def toArray2(X, y):  # collect more batches
    X_train_tmp = []
    indsTrain = []
    for song in X:
        tmp = []
        tmp2 = []
        cmpt = 0
        cmpt2 = 0
        if len(song.tracks) >= NB_TRACKS_READ:
            for track in song.tracks:
                tmp_track = []
                tmp_track2 = []
                if track > NB_NOTES_READ and cmpt < NB_TRACKS_READ:
                    for note in track.notes:
                        tmp_track.append([note.note, note.tick, note.duration])
                    cmpt += 1
                    tmp.append(tmp_track)
                    if track > 2 * NB_NOTES_READ and cmpt2 < NB_TRACKS_READ:
                        tmp_track2 += tmp_track[NB_NOTES_READ:2 * NB_NOTES_READ]
                        cmpt2 += 1
                        tmp2.append(tmp_track2)
            if cmpt == NB_TRACKS_READ:
                X_train_tmp.append(tmp)
                indsTrain.append(X.index(song))
                if cmpt2 == NB_TRACKS_READ:
                    X_train_tmp.append(tmp2)
                    indsTrain.append(X.index(song))
    return np.array([[[k for k in track[:NB_NOTES_READ]] for track in song] for song in X_train_tmp], dtype=np.float32), \
           np.array(y)[indsTrain]


X_train, y_train = toArray(X_train, y_train)
X_test, y_test = toArray(X_test, y_test)

print("X_train.shape : {0}\nX_test.shape : {1}".format(X_train.shape, X_test.shape))
print("y_train.shape : {0}\ny_test.shape : {1}".format(y_train.shape, y_test.shape))

""" useless for our data
indsTrain = np.where(np.isnan(X_train))
indsTest= np.where(np.isnan(X_test))
colMainTrain, colMainTest = np.nanmean(X_train, axis=0), np.nanmean(X_test, axis=0)
X_train[indsTrain] = np.take(colMainTrain, indsTrain[1])
X_test[indsTest] = np.take(colMainTest, indsTest[1])
"""
# y_test = extend_y(y_test)    # not in Keras
# y_train = extend_y(y_train)
# y_train, y_test = np.array(y_train),np.array(y_test)

# learning parameters
learning_rate = 0.01
epoches = 5000
batch_size = 50  # was 100. Maybe we should try to cut the files in smaller parts in order to get more samples
display_step = 10

# network parameters
n_input = 3  # was 1, changed it to get 3 information on each note
n_tracks = NB_TRACKS_READ  # for now, fixed value, but will have to be changed to "None" (any value)
n_steps = NB_NOTES_READ  # was 10
n_hidden = 500
n_classes = 2

# Keras implementation
# Model definition
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(3, 500, 3)),
    Flatten(),
    Dense(1, activation="softmax"),
])

# Choice of optimize, loss and metrics
model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Training
model.fit(X_train, y_train, epochs=200, batch_size=50)

print("Duration :", time.time() - t)