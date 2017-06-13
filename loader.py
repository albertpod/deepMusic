from cross_platform import delimiter, directory
import dataload_hex as dataload
from pathlib import Path
import joblib as jl
import os


def load_set(type, d=delimiter):
    loader = dataload.DataLoad()
    loader.reset()
    if loader.is_empty():
        loader.main(1, r'%sArchive%s%s%sjazz' % (d, d, type, d))
        loader.main(0, r'%sArchive%s%s%srap' % (d, d, type, d))
    jl.dump(loader, 'load%shex.pkl' % (type))
    return loader


if Path('loadtrainhex.pkl').is_file():
    print('Training set loading...')
    loaderTrain = jl.load('loadtrainhex.pkl')
    print('Training set loaded')
    print('Testing set loading...')
    loaderTest = jl.load('loadtesthex.pkl')
    print('Testing set loaded')
else:
    loaderTrain = load_set('train')
    loaderTest = load_set('test')
