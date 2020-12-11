from collections import Counter
from graphviz import Digraph
import math
import numpy as np
import pdb

class ID3DecisionTreeClassifier :


    def __init__(self, minSamplesLeaf = 1, minSamplesSplit = 2) :

        self.__nodeCounter = 0

        # the graph to visualise the tree
        self.__dot = Digraph(comment='The Decision Tree')

        # suggested attributes of the classifier to handle training parameters
        self.__minSamplesLeaf = minSamplesLeaf
        self.__minSamplesSplit = minSamplesSplit


    # Create a new node in the tree with the suggested attributes for the visualisation.
    # It can later be added to the graph with the respective function
    def new_ID3_node(self):
        node = {'id': self.__nodeCounter, 'value': None, 'label': None, 'attribute': None, 'entropy': None, 'samples': None,
                         'classCounts': None, 'nodes': None}

        self.__nodeCounter += 1
        return node

    # adds the node into the graph for visualisation (creates a dot-node)
    def add_node_to_graph(self, node, parentid=-1):
        nodeString = ''
        for k in node:
            if ((node[k] != None) and (k != 'nodes')):
                nodeString += "\n" + str(k) + ": " + str(node[k])

        self.__dot.node(str(node['id']), label=nodeString)
        if (parentid != -1):
            self.__dot.edge(str(parentid), str(node['id']))
            nodeString += "\n" + str(parentid) + " -> " + str(node['id'])

        print(nodeString)
        return


    # make the visualisation available
    def make_dot_data(self) :
        return self.__dot

    # inspired by https://github.com/tofti/python-id3-trees/blob/master/
    def entropy(self, n, labels):
        ent = 0
        for label in labels.keys():
            p_x = labels[label] / n
            if p_x > 0: ent += - p_x * math.log(p_x, 2)
        return ent

    # For you to fill in; Suggested function to find the best attribute to split with, given the set of
    # remaining attributes, the currently evaluated data and target.
    def find_split_attr(self, data, target, attributes, classes):
        max_info_gain = 0
        #max_info_gain_attr = None
        max_info_gain_attr = list(attributes.keys())[0]
        main_labels = Counter(target) # No. of samples in each class
        main_n = sum(main_labels.values()) # No. of samples in dataset
        main_entropy = self.entropy(main_n, main_labels) # Entropy of data with all remaining attributes.
        for ind, attr in enumerate(attributes): # Such as color, size, shape
            info_gain = main_entropy
            for e in attributes[attr]: # Attribute type: e.g. 'y', 'r', 'b'
                labels = {}
                for c in classes:
                    labels[c] = sum([1 for i, x in enumerate(data) if x[ind] == e and target[i] == c])
                n = sum(labels.values())
                if n > 0: info_gain -= (n / main_n) * self.entropy(n, labels)

            if info_gain > max_info_gain:
                max_info_gain = info_gain
                max_info_gain_attr = attr
        #print(f'{max_info_gain} - {main_entropy}')
        return max_info_gain_attr, main_entropy


    def fit(self, data, target, attributes, classes, parentid, value_prev='-'):
        root = self.new_ID3_node()
        root.update({'samples': len(target), 'value': value_prev, 'classCounts': Counter(target)})                    

        if len(root['classCounts'].keys()) == 1:
            label = target[0]
            root.update({'label': label})
            self.add_node_to_graph(root, parentid) 
            return root  

        if len(attributes) == 0:
            label = max(target, key=target.count)
            root.update({'label': label})
            self.add_node_to_graph(root, parentid)
            return root

        A, entropy = self.find_split_attr(data, target, attributes, classes)
        root.update({'attribute': A, 'entropy': entropy})
        
        # Add a subtree for each value in attribute A with max information gain.
        remaining_attributes = attributes.copy()
        remaining_attributes.pop(A)
        children = []

        value_ind = list(attributes.keys()).index(A)
        for value in attributes[A]:
            samples_v = []
            target_v = tuple()
            for ind, sample in enumerate(data): 
                if sample[value_ind] == value:
                    if type(sample) == np.ndarray: samples_v.append(np.concatenate((sample[:value_ind], sample[value_ind+1:]), axis=0))
                    else: samples_v.append(sample[:value_ind] + sample[value_ind+1:])
                    target_v = target_v + (target[ind],)

            if len(samples_v) == 0:
                leaf = self.new_ID3_node()
                label = max(target, key=target.count)
                leaf.update({'value': value, 'label': label, 'classCounts': Counter(target_v), 'samples': 0})
                self.add_node_to_graph(leaf, root['id'])
                children.append(leaf)
            else:
                subtree = self.fit(samples_v, target_v, remaining_attributes, classes, root['id'], value)
                children.append(subtree)

        root.update({'nodes': children})
        self.add_node_to_graph(root, parentid) 
        return root

    
    def predict_sample(self, node, sample, features=None):
        if node['nodes'] is None:
            return node['label']
        else:
            for child in node['nodes']:
                if child['value'] == sample[features.index(node['attribute'])]:
                    return self.predict_sample(child, sample, features)
                #else: print(f'Cannot be classified: Value: {child["value"]} - sample: {sample} - node: {features.index(node["attribute"])}')

    def predict(self, data, tree, features=None) :
        predicted = list()
        for sample in data:
            predicted_label = self.predict_sample(tree, sample, features)
            if predicted_label is None: predicted.append(0)
            else: predicted.append(predicted_label)
        return predicted