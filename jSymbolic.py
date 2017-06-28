import subprocess
import os
import xml.etree.ElementTree as ET


def get_features(midi):
    """function creates feature_values.xml file which contains features for subsequent parsing """
    # after calling this function jSymbolic will create 3 useless (in our case) files
    subprocess.call(['java', '-Xmx3048m', '-jar', 'jSymbolic2/dist/jSymbolic2.jar', midi,
                     'feature_values.xml', 'feature_descriptions.xml'])

    # jSymbolic can create csv of arff files if some special features are extracted, it is not the case here
    files = ['feature_descriptions.xml']
    os.remove(files[0])

    X = ET.parse('feature_values.xml').getroot()

    features = []
    for song in X[1:]:  # remove the header
        for feature in song[1:]:  # remove the header
            features.append(float(feature[1].text.replace(',', '.')))  # commas in XML files have to be turned into dot

    os.remove('feature_values.xml')

    # since there is only one song to extract, the output is just a list of the values of the features
    return features
