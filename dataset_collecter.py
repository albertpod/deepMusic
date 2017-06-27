import os
import numpy as np
from shutil import copyfile
from cross_platform import delimiter

dfolder = 'rapjazz'

def create_test_train_sets(split_size=.2, seed=42, d=delimiter):
    for directory in os.listdir(dfolder):
        if directory != 'jazz' and directory != 'rap': continue
        list = np.asarray(os.listdir(dfolder + delimiter + directory))
        np.random.seed(seed)
        idx_test = np.random.choice(len(list), int(len(list) * split_size))
        idx_train = np.setdiff1d(np.arange(0, len(list)), idx_test)
        test_set = list[idx_test]
        train_set = list[idx_train]
        [[copyfile(dfolder + d + directory + d + x, dfolder + d + set_type + d + directory + d + x)
          for x in set_list] for set_type, set_list in zip(['test', 'train'], [test_set, train_set])]


create_test_train_sets()
