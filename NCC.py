import numpy as np
from numpy.linalg import norm

class NearestCentroidClassifier:

    def __init__(self):
        self.centroids = [] # Abandoned dictionary and used list. We are now sorting the classes anyways in fit.


    def calculate_centroid(self, features, label):
        # Since features are concatenated columns, we have a 1-dimensional array.
        num_attrs = features.shape[1] # or len(features[0])
        class_centroid = np.zeros(num_attrs)
        for attr in range(num_attrs):
            mean_value_for_attr = np.mean([feature[attr] for feature in features])
            class_centroid[attr] = mean_value_for_attr
        self.centroids.append(class_centroid)


    def fit(self, features, labels):
        classes = sorted(list(set(labels)))
        for c in classes:
            class_indices = np.where(labels == c)[0] # Numpy array in first element of tuple returned by np.where()
            class_features = np.take(features, class_indices, axis=0)
            self.calculate_centroid(class_features, c)


    def predict(self, features):
        # Implementation based on a list of features and labels. for single predictions simple modifications have to be made.
        predicted_labels = []
        for feature in features:
            distances = [norm(feature - centroid) for centroid in self.centroids]
            nearest_label = np.argmin(distances)
            predicted_labels.append(nearest_label)
        return predicted_labels

    def get_centroids(self):
        return self.centroids