# -*- coding: utf-8 -*-
"""
Created on Wed May 31 11:15:17 2017

@author: Theo
"""

#from __future__ import print_function

import tensorflow as tf
import joblib as jl
from tensorflow.contrib import rnn
import numpy as np
import os
import dataload
import time
import prettytensor as pt
from main import loaderTrain, loaderTest
#from tensorflow.examples.tutorials.mnist import input_data
#mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)


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

#dataset = jl.load(r'C:\Users\Theo\Desktop\ClassifierAlbert\datasetSeries.pkl')
#X_train, X_test, y_train, y_test = dataset

X_train, X_test, y_train, y_test = loaderTrain.songs[:],loaderTest.songs[:],loaderTrain.artists[:],loaderTest.artists[:]
# for now, X_train is not gonna be an np.array, it will take the good shape later

def toArray(X, y):
    X_train_tmp=[]
    indsTrain =[]
    for song in X:
        tmp = []
        cmpt = 0
        if len(song.tracks)>2:
            for track in song.tracks:
                tmp_track = []
                if track>NB_NOTES_READ and cmpt < 3:
                    for note in track.notes:
                        tmp_track.append([note.note, note.tick, note.duration])
                    cmpt +=1
                    tmp.append(tmp_track)
            if cmpt == NB_TRACKS_READ:
                X_train_tmp.append(tmp)
                indsTrain.append(X.index(song))
            cmpt =0
    return np.array([[[k for k in track[:NB_NOTES_READ]] for track in song] for song in X_train_tmp],dtype=np.float32),np.array(y)[indsTrain]
            
X_train, y_train = toArray(X_train,y_train)
X_test,y_test = toArray(X_test,y_test)

#%%
print("X_train.shape : {0}\nX_test.shape : {1}".format(X_train.shape,X_test.shape))
print("y_train.shape : {0}\ny_test.shape : {1}".format(y_train.shape,y_test.shape))

""" useless for our data
indsTrain = np.where(np.isnan(X_train))
indsTest= np.where(np.isnan(X_test))
colMainTrain, colMainTest = np.nanmean(X_train, axis=0), np.nanmean(X_test, axis=0)
X_train[indsTrain] = np.take(colMainTrain, indsTrain[1])
X_test[indsTest] = np.take(colMainTest, indsTest[1])
"""
y_test = extend_y(y_test)
y_train = extend_y(y_train)
#y_train, y_test = np.array(y_train),np.array(y_test)

# learning parameters
learning_rate = 0.001
epoches = 5000
batch_size = 20 #was 100. Maybe we should try to cut the files in smaller parts in order to get more samples
display_step = 10

# network parameters
n_input = 3 #was 1, changed it to get 3 informations on each note
n_tracks = NB_TRACKS_READ #for now, fixed value, but will have to be changed to "None" (any value)
n_steps = NB_NOTES_READ #was 10
n_hidden = 50
n_classes = 2

# tf graph placeholders
with tf.variable_scope('test10'):
    X = tf.placeholder(shape=[None,n_tracks, n_steps, n_input], dtype=tf.float32)
    y = tf.placeholder(shape=[None, n_classes], dtype=tf.float32)
    
    # define weights
    weights = tf.Variable(tf.random_normal([n_hidden, n_classes]))
    biases = tf.Variable(tf.random_normal([n_classes]))
    
    def RNN(X, weights, biases):
        # unstack to get a list of 'n_steps' tensors of shape (batch_size, n_input)
        X = tf.unstack(X, n_tracks,1)
        for k in range(len(X)):
            X[k] = tf.unstack(X[k], n_steps, 1)
        X = X[0]+X[1]+X[2]
        lstm_cell = rnn.BasicLSTMCell(n_hidden)

        # lstm cell output
        outputs, states = rnn.static_rnn(lstm_cell, X, dtype=tf.float32)
        #activation_output = tf.reshape(tf.matmul(outputs[-1], weights) + biases, (batch_size,))
        return tf.matmul(outputs[-1], weights) + biases
    
    pred = RNN(X, weights, biases)
    
    # define loss and optimizer
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)
    
    # evaluate model
    correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    
    # initialize
    init = tf.global_variables_initializer()
    
    # launch the graph
    with tf.Session() as sess:
        sess.run(init)
        step = 1
        # keep training till max iterations
        while step * batch_size < epoches:
            rand_index = np.random.choice(len(X_train), size=batch_size)
            rand_x = X_train[rand_index]
            rand_y = y_train[rand_index]
            rand_x = rand_x.reshape((batch_size,n_tracks, n_steps, n_input))
            rand_y = rand_y.reshape((batch_size, n_classes)) # why y has this shape ?
            rand_y = rand_y.astype(float)
            sess.run(optimizer, feed_dict={X: rand_x, y: rand_y})
            sess.run(pred, feed_dict={X: rand_x, y: rand_y})
            #predicted, real = pred.eval(feed_dict={X: rand_x, y: rand_y}).reshape((100,)), rand_y.reshape((100,))
            #softmax_xentropy = tf.nn.softmax_cross_entropy_with_logits(logits=predicted, labels=real)
            #xentropy_softmax_y_out = sess.run(softmax_xentropy)
            if step % display_step == 0:
                acc = sess.run(accuracy, feed_dict={X: rand_x, y: rand_y})
                temp_loss = sess.run(loss, feed_dict={X: rand_x, y: rand_y})
                print('Generation: %s. Loss = %s, Accuracy = %s' % (step + 1, temp_loss, acc))
            step += 1
        print('Optimization finished')
    
        X_test = X_test.reshape((len(X_test),n_tracks, n_steps, n_input))
        y_test = y_test.reshape((len(y_test), n_classes))
    
        print("Testing Accuracy :",  sess.run(accuracy, feed_dict={X: X_test, y: y_test}))
        print("Duration :",time.time()-t)