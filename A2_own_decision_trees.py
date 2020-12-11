import ToyData as td
import ID3

import numpy as np
from sklearn import tree, metrics, datasets
from sklearn.model_selection import train_test_split

from collections import OrderedDict
from graphviz import Source


def main():

    attributes, classes, data, target, data2, target2 = td.ToyData().get_data()

    id3 = ID3.ID3DecisionTreeClassifier()

    #myTree = id3.fit(data, target, attributes, classes, parentid=-1)
    #print(myTree)
    #plot = id3.make_dot_data()
    #plot.render("testTree")
    #predicted = id3.predict(data2, myTree)
    #print(predicted)

    digits = datasets.load_digits()
    X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target, train_size=0.7, random_state=0)
    attributes = OrderedDict()
    
    improved = False

    # Create attribute dictionary
    for index, attr in enumerate(digits.feature_names):
        if improved: attributes[attr] = ['dark', 'grey', 'light']
        else: attributes[attr] = list(range(0, 17))
    
    tuple_y_train = tuple()
    # Represent targets in a tuple
    for label in y_train:
        tuple_y_train += (label,)
    
    tuple_y_test = tuple()
    for label in y_test:
        tuple_y_test += (label,)
    
    if improved:
        X_train_improved = []
        X_test_improved = []
        for sample in X_train:
            new_sample = []
            for value in sample:
                if value < 5:
                    new_sample.append('dark')
                elif value < 10:
                    new_sample.append('grey')
                else: new_sample.append('light')
            X_train_improved.append(new_sample)

        for sample in X_test:
            new_sample = []
            for value in sample:
                if value < 5:
                    new_sample.append('dark')
                elif value < 10:
                    new_sample.append('grey')
                else: new_sample.append('light')
            X_test_improved.append(new_sample)
    
    if not improved:
        tree = id3.fit(X_train, tuple_y_train, attributes, digits.target_names, parentid=-1)
        predicted = id3.predict(X_test, tree, digits.feature_names)
    else:
        tree = id3.fit(X_train_improved, tuple_y_train, attributes, digits.target_names, parentid=-1)
        predicted = id3.predict(X_test_improved, tree, digits.feature_names)
    evaluate(tuple_y_test, predicted)

    plot = id3.make_dot_data()
    plot.render("digitsTree")


def evaluate(true, pred):
    print("Classification report for classifier %s:\n%s\n"
      % ('ID3', metrics.classification_report(true, pred)))
    print("Confusion matrix:\n%s" % metrics.confusion_matrix(true, pred))



if __name__ == "__main__": main()