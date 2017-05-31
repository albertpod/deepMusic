# -*- coding: utf-8 -*-
"""
Created on Tue May 30 12:04:19 2017

@author: Theo
"""
"""Coordinate everything
"""

import os
import dataload

os.chdir(r'C:\Users\Theo\Desktop\ClassifierAlbert\data')

#%% cellule

loaderTrain = dataload.DataLoad()
loaderTrain.reset()
if loaderTrain.is_empty():
    loaderTrain.main(1,r'\Train\Led Zeppelin')
    loaderTrain.main(0,r'\Train\Bach')
    print("Training set importer")

loaderTest = dataload.DataLoad()
loaderTest.reset()
if loaderTest.is_empty():
    loaderTest.main(1,r'\Test\Led Zeppelin')
    loaderTest.main(0,r'\Test\Bach')
    print("Testing set imported")