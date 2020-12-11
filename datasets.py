from sklearn import datasets
from sklearn.model_selection import train_test_split
import MNIST

import numpy as np

def sklearn_digits(normalized=True, shuffle=True):
    digits = datasets.load_digits()
    X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target, train_size=0.7, random_state=0, shuffle=shuffle)
    if normalized == True: return X_train / 16, X_test / 16, y_train, y_test
    else: return X_train, X_test, y_train, y_test


def sklearn_digits_summarized():
    digits = datasets.load_digits()
    X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target, train_size=0.7, random_state=0)
    
    X_train_summarized = []
    X_test_summarized = []
    for sample in X_train:
        new_sample = []
        for value in sample:
            if value < 5: # Dark
                new_sample.append(0)
            elif value < 10: # Grey
                new_sample.append(1)
            else: # Light
                new_sample.append(2)
        X_train_summarized.append(new_sample)

    for sample in X_test:
        new_sample = []
        for value in sample:
            if value < 5: # Dark
                new_sample.append(0)
            elif value < 10: # Grey
                new_sample.append(1)
            else: # Light
                new_sample.append(2)
        X_test_summarized.append(new_sample)

    return np.asarray(X_train_summarized), np.asarray(X_test_summarized), y_train, y_test


def MNIST_light(normalized=True):
    mnist = MNIST.MNISTData('MNIST_Light/*/*.png')
    train_features, test_features, train_labels, test_labels = mnist.get_data(normalized)
    return train_features, test_features, train_labels, test_labels