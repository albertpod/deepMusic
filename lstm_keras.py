import tensorflow as tf
from tensorflow.contrib import rnn
import numpy as np
import os
import dataload
import time
from loader import loaderTrain, loaderTest
import keras as keras
from keras.models import Sequential,Model
from keras.layers import Dense,Activation,LSTM,Conv1D,Conv2D,Flatten,Embedding,Reshape,Input,ConvLSTM2D

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

def toArrayTracks(X,y):
    """Turns lists X and y into arrays, only using relevant informations and a certain number of tracks and notes.
        This function selects the track that starts the latest and put the temporal starting point of gathering the input
        there. This function returns each track separately.
        Params:
            X: cf dataload
            y: cf dataload
        Return:
             (X1,X2,X3): array of shape (number of songs kept, NB_NOTES_READ, 3)
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
                        if note.tick >= start_tick:
                            tmp_track.append([note.note, note.tick, note.duration])
                    cmpt += 1
                    tmp.append(tmp_track)
            if cmpt == NB_TRACKS_READ:
                X_train_tmp.append(tmp)
                indsTrain.append(X.index(song))
    return np.array([[k for k in song[0][:NB_NOTES_READ]] for song in X_train_tmp], dtype=np.float32), \
           np.array([[k for k in song[1][:NB_NOTES_READ]] for song in X_train_tmp], dtype=np.float32), \
           np.array([[k for k in song[2][:NB_NOTES_READ]] for song in X_train_tmp], dtype=np.float32), \
           np.array(y)[indsTrain]


'''X_train, y_train = toArray(X_train, y_train)
X_test, y_test = toArray(X_test, y_test)'''

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
'''
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(3, 500, 3)),
    LSTM(128),
    Flatten(),
    Dense(256, activation='relu'),
    Dense(1, activation="softmax"),
]
'''

# This model divides songs into 3 tracks and performs Conv1D on each track, then LSTM. Does not provide good results...
X_train1, X_train2, X_train3, y_train = toArrayTracks(X_train, y_train)
X_test1 , X_test2, X_test3, y_test = toArrayTracks(X_test, y_test)

print("X_train.shape : {0}\nX_test.shape : {1}".format(X_train1.shape, X_test1.shape))
print("y_train.shape : {0}\ny_test.shape : {1}".format(y_train.shape, y_test.shape))

input1 = Input(shape=(500,3,), dtype='float32', name='input1')
input2 = Input(shape=(500,3,), dtype='float32', name='input2')
input3 = Input(shape=(500,3,), dtype='float32', name='input3')

ins = [input1,input2,input3]

for k in range(3):   # For each track, performs the following structure
    ins[k] = Conv1D(64, 6, activation="relu")(ins[k])
    ins[k] = Conv1D(128, 3, activation="relu")(ins[k])
    ins[k] = LSTM(64, return_sequences=True)(ins[k])
    ins[k] = Conv1D(64, 6, activation="relu")(ins[k])
    ins[k] = Conv1D(128, 3, activation="relu")(ins[k])
    ins[k] = LSTM(128)(ins[k])

# Merges the tracks
x = keras.layers.concatenate(ins)

# Dense (aka Fully connected) 3 times in a row
for k in range(3):
    x = Dense(128,activation='relu')(x)

# Output
main_output = Dense(1,activation="softmax",name="main_output")(x)

# Writing the model
model = Model(inputs=[input1, input2, input3], outputs=[main_output])

# Custom optimizer
sgd = keras.optimizers.SGD(lr=0.1, decay=1e-6, momentum=1.9)

# Choice of optimizer, loss and metrics
model.compile(optimizer=sgd,
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Training for track by track conv1D
model.fit([X_train1, X_train2, X_train3], y_train, epochs=20, batch_size=50)

# Training
# model.fit(X_train, y_train, epochs=20, batch_size=50)

# Testing for track by track conv1D
score = model.evaluate([X_test1,X_test2,X_test3], y_test, batch_size=32)

# Testing
#score = model.evaluate(X_test, y_test, batch_size=32)

print("\nScore :", score)
print("\nDuration :", time.time() - t)