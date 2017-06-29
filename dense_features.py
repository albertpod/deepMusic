import os
import time
import xml.etree.ElementTree as ET
import keras
import matplotlib.pyplot as plt
from matplotlib import offsetbox
import matplotlib.patheffects as PathEffects
import seaborn as sns
import numpy as np
import tensorflow as tf
from keras.layers import Dense, Dropout
from keras.models import Sequential
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from deferred import dataload_hex
from sklearn import datasets
#from XGBoost import run_XGBoost

NB_NOTES_READ = dataload_hex.MIN_SIZE

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
print('TensorFlow version: {0}'.format(tf.__version__))

t = time.time()

X_train1 = ET.parse(
    r"jSymbolic2\features\extracted_feature_values_train_jazz.xml").getroot()
X_train2 = ET.parse(
    r"jSymbolic2\features\extracted_feature_values_train_rap.xml").getroot()
X_test1 = ET.parse(
    r"jSymbolic2\features\extracted_feature_values_test_jazz.xml").getroot()
X_test2 = ET.parse(
    r"jSymbolic2\features\extracted_feature_values_test_rap.xml").getroot()

print("XML files parsed")


def parse(X, y):
    out = []
    for song in X[1:]:  # remove the first
        features = []
        for feature in song[1:]:  # remove the first
            features.append(float(feature[1].text.replace(',', '.')))  # commas in XML files have to be turned into dot
        out.append(features)
    return out, [y for k in range(len(X[1:]))]


def data_process(X_train1, X_train2, X_test1, X_test2):
    xt1, yt1 = parse(X_train1, 1)
    xt2, yt2 = parse(X_train2, 0)
    X_train = np.array(xt1 + xt2)
    y_train = np.array(yt1 + yt2)

    xt1, yt1 = parse(X_test1, 1)
    xt2, yt2 = parse(X_test2, 0)
    X_test = np.array(xt1 + xt2)
    y_test = np.array(yt1 + yt2)

    X_tot = np.concatenate((X_test, X_train))
    y_tot = np.concatenate((y_test, y_train))

    # Standardization
    def standardize(X):
        scaler = StandardScaler().fit(X)
        rescaled = scaler.transform(X)
        return rescaled

    X_tot = standardize(X_tot)

    # Randomization
    inds = np.random.permutation(X_tot.shape[0])
    train_inds, test_inds = inds[:int(0.8 * len(inds))], inds[int(0.8 * len(inds)):]  # 80% training set, 20% testing set
    X_train = X_tot[train_inds]
    X_test = X_tot[test_inds]
    y_train = y_tot[train_inds]
    y_test = y_tot[test_inds]

    return X_train, X_test, y_train, y_test

X_train, X_test, y_train, y_test = data_process(X_train1, X_train2, X_test1, X_test2)

# GBM
def call_GMB():
    clf = GradientBoostingClassifier()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print("GBM accuracy = ", accuracy, "Precision = ", precision_score(y_test, y_pred))

    print("X_train.shape : {0}\nX_test.shape : {1}".format(X_train.shape, X_test.shape))
    print("y_train.shape : {0}\ny_test.shape : {1}".format(y_train.shape, y_test.shape))

def tSNE(X, y, labels):
    model = TSNE(n_components=3, random_state=42)
    np.set_printoptions(suppress=True)
    tX = model.fit_transform(X)
    # We choose a color palette with seaborn.
    palette = np.array(sns.color_palette("hls", 2))

    # We create a scatter plot.
    f = plt.figure(figsize=(8, 8))
    ax = plt.subplot(aspect='equal')
    sc = ax.scatter(tX[:, 0], tX[:, 1], lw=0, s=40,
                    c=palette[y.astype(np.int)])
    plt.xlim(-25, 25)
    plt.ylim(-25, 25)
    ax.axis('off')
    ax.axis('tight')

    # We add the labels for each digit.
    txts = []
    for label in labels:
        # Position of each label.
        g = tX[y == labels.index(label), :]
        xtext, ytext = np.median(tX[y == labels.index(label), :], axis=0)
        txt = ax.text(xtext, ytext, label, fontsize=24)
        txt.set_path_effects([
            PathEffects.Stroke(linewidth=5, foreground="w"),
            PathEffects.Normal()])
        txts.append(txt)
    plt.show()
    return f, ax, sc, txts

tSNE(X_train, y_train, ['jazz', 'rap'])
#call_GMB()

# learning parameters
learning_rate = 0.001
epoches = 10000
batch_size = 100
dropout = 0.5
dense_layers = 600


def run_model():
    # 1st model : Dense layers
    model = Sequential([
        Dense(dense_layers, activation="relu", input_shape=(156,)),
        Dropout(dropout),
        Dense(dense_layers, activation="relu"),
        Dropout(dropout),
        Dense(dense_layers, activation="relu"),
        Dropout(dropout),
        Dense(300, activation="relu"),
        Dropout(dropout),
        Dense(1, activation="sigmoid")
    ])

    # Custom optimizer
    sgd = keras.optimizers.SGD(lr=learning_rate)

    # Choice of optimizer, loss and metrics
    model.compile(optimizer=sgd,
                  loss='binary_crossentropy',
                  metrics=['binary_accuracy'])

    # Training
    hist1 = model.fit(X_train, y_train, epochs=int(epoches * 2 / 3), batch_size=batch_size, verbose=2)
    model.optimizer.lr = tf.constant(learning_rate / 10)
    hist2 = model.fit(X_train, y_train, epochs=int(epoches / 3), batch_size=batch_size, verbose=2)

    # Testing
    score = model.evaluate(X_test, y_test)

    print("\nScore :", score)
    print("\nDuration :", time.time() - t)

    # Saving the model
    model.save(r"models/dense_xml_jazz_rap_%.3f.h5" % score[1])

    plt.plot(hist1.history.get("binary_accuracy") + hist2.history.get("binary_accuracy"))
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.axis([0, epoches, 0, 1])
    plt.savefig(r'plot/Dense_4layers%s_%.1fdropout_%sbatch_%.3ftest.png' % (dense_layers, dropout, batch_size, score[1]))

run_model()
