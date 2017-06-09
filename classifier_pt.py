# -*- coding: utf-8 -*-
"""
Created on Wed May 31 11:15:17 2017

@author: Theo
"""

# from __future__ import print_function

import tensorflow as tf
import joblib as jl
from tensorflow.contrib import rnn
import numpy as np
import os
import dataload
import time
import prettytensor as pt
from loader import loaderTrain, loaderTest

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

X_train, X_test, y_train, y_test = loaderTrain.songs[:], loaderTest.songs[:], loaderTrain.artists[:], loaderTest.artists[:]


def toArray(X, y):
    """Turns lists X and y into arrays, only using relevant informations and a certain number of tracks and notes.
    This function selects the track that starts the latest and put the temporal starting point of gathering the input
    there.
    Args:
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
y_test = extend_y(y_test)
y_train = extend_y(y_train)
# y_train, y_test = np.array(y_train),np.array(y_test)

# learning parameters
learning_rate = 0.001
epoches = 5000
batch_size = 50  # was 100. Maybe we should try to cut the files in smaller parts in order to get more samples
display_step = 10

# network parameters
n_input = 3  # was 1, changed it to get 3 information on each note
n_tracks = NB_TRACKS_READ  # for now, fixed value, but will have to be changed to "None" (any value)
n_steps = NB_NOTES_READ  # was 10
n_hidden = 50
n_classes = 2

# Placeholders
X = tf.placeholder(shape=[None, n_tracks, n_steps, n_input], dtype=tf.float32, name="X")
y_true = tf.placeholder(shape=[None, n_classes], dtype=tf.float32, name="y_true")
y_true_cls = tf.argmax(y_true, dimension=1)

# Pretty Tensor implementation
x_pretty = pt.wrap(X)

with pt.defaults_scope(activation_fn=tf.nn.relu):
    y_pred, cost = x_pretty.\
        conv2d(kernel=20, depth=16, name='layer_conv1').\
        max_pool(kernel=2, stride=2).\
        conv2d(kernel=10, depth=32, name='layer_conv2').\
        max_pool(kernel=2, stride=2).\
        dropout(0.9). \
        flatten().\
        fully_connected(size=32, name='layer_fc1').\
        softmax_classifier(num_classes=n_classes, labels=y_true)

    y_pred_cls = tf.argmax(y_pred, dimension=1)

    optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(cost)
    correct_prediction = tf.equal(y_pred_cls, y_true_cls)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    session = tf.Session()
    session.run(tf.global_variables_initializer())
    train_batch_size = 50

    # Counter for total number of iterations performed so far.
    total_iterations = 0


def optimize(num_iterations):
    # Ensure we update the global variable rather than a local copy.
    global total_iterations

    # Start-time used for printing time-usage below.
    start_time = time.time()

    for i in range(total_iterations,
                   total_iterations + num_iterations):

        # Get a batch of training examples.
        # x_batch now holds a batch of images and
        # y_true_batch are the true labels for those images.
        rand_index = np.random.choice(len(X_train), size=batch_size)
        x_batch = X_train[rand_index]
        y_true_res = y_train[rand_index]
        x_batch = x_batch.reshape((batch_size, n_tracks, n_steps, n_input))
        y_true_res = y_true_res.reshape((batch_size, n_classes))
        y_true_res = y_true_res.astype(float)

        # Put the batch into a dict with the proper names
        # for placeholder variables in the TensorFlow graph.
        feed_dict_train = {X: x_batch,
                           y_true: y_true_res}

        # Run the optimizer using this batch of training data.
        # TensorFlow assigns the variables in feed_dict_train
        # to the placeholder variables and then runs the optimizer.
        session.run(optimizer, feed_dict=feed_dict_train)

        # Print status every 100 iterations.
        if i % 100 == 0:
            # Calculate the accuracy on the training-set.
            acc = session.run(accuracy, feed_dict=feed_dict_train)

            # Message for printing.
            msg = "Optimization Iteration: {0:>6}, Training Accuracy: {1:>6.1%}"

            # Print it.
            print(msg.format(i + 1, acc))

    # Update the total number of iterations performed.
    total_iterations += num_iterations

    # Ending time.
    end_time = time.time()

    # Difference between start and end-times.
    time_dif = end_time - start_time

    X_tes = X_test.reshape((len(X_test), n_tracks, n_steps, n_input))
    y_tes = y_test.reshape((len(y_test), n_classes))

    print("Testing Accuracy :", session.run(accuracy, feed_dict={X: X_tes, y_true: y_tes}))


optimize(num_iterations=3000)

print("Duration :", time.time() - t)
