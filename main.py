# -*- coding: utf-8 -*-
"""
Created on Tue May 30 12:04:19 2017

@authors: Theo, Albert

Coordinate everything
"""

from cross_platform import delimiter, directory
import dataload

def load_set(type):
    loader = dataload.DataLoad()
    loader.reset()
    if loader.is_empty():
        [loader.main(i, r'%sdata%s%s%sLed Zeppelin' % (delimiter, delimiter, type, delimiter)) for i in range(2)]
    return loader

loaderTrain = load_set('Train')
loaderTest = load_set('Test')