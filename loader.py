from cross_platform import delimiter, directory
import dataload
from pathlib import Path
import joblib as jl


def load_set(type, d=delimiter):
    loader = dataload.DataLoad()
    loader.reset()
    if loader.is_empty():
        loader.main(1, r'%sdataset_jr%s%s%sjazz' % (d, d, type, d))
        loader.main(0, r'%sdataset_jr%s%s%rock' % (d, d, type, d))
    jl.dump(loader, 'load%s.pkl' % (type))
    return loader


if Path('loadTrain.pkl').is_file():
    loaderTrain = jl.load('loadTrain.pkl')
    loaderTest = jl.load('loadTest.pkl')
else:
    loaderTrain = load_set('train')
    loaderTest = load_set('test')
