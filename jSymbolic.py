import subprocess
import os

def get_features(midi):
    """function creates feature_values.xml file which contains features for subsequent parsing """
    # after calling this function jSymbolic will create 3 useless (in our case) files
    subprocess.call(['java', '-Xmx3048m', '-jar', 'jSymbolic2/jSymbolic2.jar', midi,
                     'feature_values.xml', 'feature_descriptions.xml'])

    files = ['feature_descriptions.xml', 'feature_values.csv', 'feature_values.arff']
    for file in files: os.remove(file)

    # TODO: parse the features from feature_values.xml here

    os.remove('feature_values.xml')

    # TODO: return parsed object