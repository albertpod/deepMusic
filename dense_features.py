import tensorflow as tf
from tensorflow.contrib import rnn
import numpy as np
import os
import dataload_hex
import time
import keras
from keras.models import Sequential,Model
from keras.layers import Dense,Activation,LSTM,Conv1D,Conv2D,Flatten,Embedding,Reshape,Input,ConvLSTM2D,\
    TimeDistributed,Dropout,MaxPooling1D,GlobalAveragePooling1D
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score
from sklearn.preprocessing import StandardScaler

NB_NOTES_READ = dataload_hex.MIN_SIZE

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
print('TensorFlow version: {0}'.format(tf.__version__))

t = time.time()

X_train1 = ET.parse(r"C:\Users\Theo\Desktop\jMIR_3_0_developer\jMIR_3_0_developer\jSymbolic2\dist\extracted_feature_values_train_jazz.xml").getroot()
X_train2 = ET.parse(r"C:\Users\Theo\Desktop\jMIR_3_0_developer\jMIR_3_0_developer\jSymbolic2\dist\extracted_feature_values_train_rap.xml").getroot()
X_test1 = ET.parse(r"C:\Users\Theo\Desktop\jMIR_3_0_developer\jMIR_3_0_developer\jSymbolic2\dist\extracted_feature_values_test_jazz.xml").getroot()
X_test2 = ET.parse(r"C:\Users\Theo\Desktop\jMIR_3_0_developer\jMIR_3_0_developer\jSymbolic2\dist\extracted_feature_values_test_rap.xml").getroot()

print("XML files parsed")

def toArray(X, y):
    out = []
    for song in X[1:]:   # remove the first
        features = []
        for feature in song[1:]:  # remove the first
            features.append(float(feature[1].text.replace(',', '.')))   # commas in XML files have to be turned into dot
        out.append(features)
    return out, [y for k in range(len(X[1:]))]

xt1, yt1 = toArray(X_train1, 1)
xt2, yt2 = toArray(X_train2, 0)
X_train = np.array(xt1 + xt2)
y_train = np.array(yt1 + yt2)

xt1, yt1 = toArray(X_test1, 1)
xt2, yt2 = toArray(X_test2, 0)
X_test = np.array(xt1 + xt2)
y_test = np.array(yt1 + yt2)

# GBM
clf = GradientBoostingClassifier()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("GBM accuracy = ", accuracy, "Precision = ", precision_score(y_test, y_pred))

print("X_train.shape : {0}\nX_test.shape : {1}".format(X_train.shape, X_test.shape))
print("y_train.shape : {0}\ny_test.shape : {1}".format(y_train.shape, y_test.shape))

def standardize(X):
    scaler = StandardScaler().fit(X)
    rescaled = scaler.transform(X)
    return rescaled

X_train = standardize(X_train)
X_test = standardize(X_test)

# learning parameters
learning_rate = 0.001
epoches = 10000
batch_size = 100
dropout = 0.6
dense_layers = 600

'''# 1st model : LSTM
model = Sequential([
    LSTM(128, return_sequences=True,input_shape=(156, 1,)),
    Dropout(0.25),
    LSTM(128,return_sequences=False),
    Dropout(0.25),
    Dense(128,activation="relu"),
    Dense(1, activation="sigmoid")
])'''

# 2nd model : Dense layers
model = Sequential([
    Dense(dense_layers, activation="relu", input_shape=(156,)),
    Dropout(dropout),
    Dense(dense_layers, activation="relu"),
    Dropout(dropout),
    Dense(dense_layers, activation="relu"),
    Dropout(dropout),
    Dense(300,activation="relu"),
    Dense(1, activation="sigmoid")
])

# Custom optimizer
sgd = keras.optimizers.SGD(lr=learning_rate)

# Choice of optimizer, loss and metrics
model.compile(optimizer=sgd,
              loss='binary_crossentropy',
              metrics=['binary_accuracy'])

# Training
hist1 = model.fit(X_train, y_train, epochs=int(epoches*2/3), batch_size=batch_size, verbose=2)
model.optimizer.lr = tf.constant(0.0001)
hist2 = model.fit(X_train, y_train, epochs=int(epoches/3), batch_size=batch_size, verbose=2)

# Testing
score = model.evaluate(X_test, y_test, batch_size=32)

print("\nScore :", score)
print("\nDuration :", time.time() - t)

# Saving the model
model.save("dense_xml_jazz_rap.h5")

plt.plot(hist1.history.get("binary_accuracy") + hist2.history.get("binary_accuracy"))
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.axis([0, epoches, 0, 1])
plt.savefig(r'plot/Dense_4layers%s_%.1fdropout_%sbatch_%.3ftest.png' % (dense_layers, dropout, batch_size, score[1]))