import numpy as np
from scipy.stats import norm
from tqdm import tqdm

class NaiveBayesianClassifier:

    def __init__(self):
        self.class_features = {}
        self.class_priors = {}
        self.classes = []


    def fit(self, features, labels):
        self.classes = sorted(list(set(labels)))
        for c in self.classes:
            class_indices = np.where(labels == c)[0] # Numpy array in first element of tuple returned by np.where()
            self.class_features[c] = np.take(features, class_indices, axis=0)
            self.class_priors[c] = len(self.class_features) / len(features)

    def calculate_posterior(self, feature, label):
        likelihood = self.class_priors[label]
        num_attrs = len(feature)
        for attr in range(num_attrs): # For every pixel in image
            likelihood *= len([train_feat[attr] for train_feat in self.class_features[label] if train_feat[attr] == feature[attr]]) / len(self.class_features)
        return likelihood


    def predict(self, features):
        predicted_labels = []
        for feat in tqdm(features):
            predicted_label = np.argmax([self.calculate_posterior(feat, c) for c in self.classes])
            predicted_labels.append(predicted_label)
        return predicted_labels

class GaussianNaiveBayesianClassifier:

    def __init__(self):
        self.class_features = {}
        self.class_priors = {}
        self.gaussian_info = {}
        self.classes = []

    def calculate_gaussian(self, num_attrs, label, epsilon):
        attr_info = []
        for attr in range(num_attrs): # For every pixel in image
            # Mean, standard deivation for pixel in all features
            attr_mean = np.mean([feature[attr] for feature in self.class_features[label]])
            attr_std = np.std([feature[attr] for feature in self.class_features[label]])
            if int(attr_std) == 0: attr_std += epsilon
            attr_info.append((attr_mean, attr_std))
        return attr_info

    def fit(self, features, labels, epsilon=0):
        self.classes = sorted(list(set(labels)))
        for c in self.classes:
            class_indices = np.where(labels == c)[0] # Numpy array in first element of tuple returned by np.where()
            self.class_features[c] = np.take(features, class_indices, axis=0)
            self.class_priors[c] = len(self.class_features[c]) / len(features)
            self.gaussian_info[c] = self.calculate_gaussian(features.shape[1], c, epsilon)

    def calculate_posterior(self, feature, label):
        attr_info = self.gaussian_info[label]
        likelihood = self.class_priors[label]
        for attr in range(len(attr_info)): # For every pixel in image
            loc = attr_info[attr][0]
            std = attr_info[attr][1]
            likelihood *= norm.pdf(feature[attr], loc=loc, scale=std) # log(a*b) = log(a) + log(b)
        return likelihood


    def predict(self, features):
        predicted_labels = []
        for feat in tqdm(features):
            predicted_label = np.argmax([self.calculate_posterior(feat, c) for c in self.classes])
            predicted_labels.append(predicted_label)
        return predicted_labels