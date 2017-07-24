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
from sklearn.model_selection import StratifiedKFold
from deferred import dataload_hex
from sklearn import datasets
from mpl_toolkits.mplot3d import Axes3D
#from XGBoost import run_XGBoost

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
print('TensorFlow version: {0}'.format(tf.__version__))

X1 = ET.parse(
    r"jSymbolic2\features\extracted_feature_blues_feat.xml").getroot()
X2 = ET.parse(
    r"jSymbolic2\features\extracted_feature_random_feat.xml").getroot()
X3 = ET.parse(
    r"jSymbolic2\features\extracted_feature_mid_test.xml").getroot()

model_path ="models/dense_xml_blues_random_1.000.h5"


def parse(X, y):
    out = []
    for song in X[1:]:  # remove the first
        features = []
        for feature in song[1:]:  # remove the first
            features.append(float(feature[1].text.replace(',', '.')))  # commas in XML files have to be turned into dot
        out.append(features)
    return out, [y for k in range(len(X[1:]))]


def get_data(X1, X2, X3=None, X4=None):
    xt1, yt1 = parse(X1, 0)
    xt2, yt2 = parse(X2, 1)
    xt3, yt3, xt4, yt4 = None, None, None, None
    if X3 != None:
        xt3, yt3 = parse(X3, 2)
    if X4 != None:
        xt4, yt4 = parse(X4, 3)

    X_tot = np.concatenate(tuple((k for k in (xt1, xt2, xt3, xt4) if k is not None)))
    y_tot = np.concatenate(tuple((k for k in (yt1, yt2, yt3, yt4) if k is not None)))
    return X_tot, y_tot


# Standardization with regard to the original data is needed in order to have the same normalization

X_tot, y_tot = get_data(X1, X2)
scaler = StandardScaler().fit(X_tot)

model = keras.models.load_model(model_path)


def fitness(features):

    X_test = scaler.transform(features)

    hist = model.predict(X_test, verbose=0)
    return [hist[k][0] for k in range(len(hist))]

'''verif = [k[0]>0.5 for k in hist]

print("\nScore :", verif.count(True)/len(verif))

for k in range(len(verif)):
    if not verif[k]:
        print(X3[k][0].text)
        print("Confidence :", hist[k][0]*100, "%")'''