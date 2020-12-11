import numpy as np
from scipy.stats import norm as normal
from numpy.linalg import norm
from tqdm import tqdm
import matplotlib.pyplot as plt


class GaussianEM:

    def __init__(self, num_clusters=10, epsilon=1e-3, threshold=1e-3):
        self.cluster_features = {}
        self.cluster_priors = {}
        self.cluster_center_means = {}
        self.gaussian_info = {}
        self.num_clusters = num_clusters
        self.epsilon = epsilon
        self.threshold = threshold


    def probability(self, feature, cluster):
        attr_info = self.gaussian_info[cluster]
        likelihood = self.cluster_priors[cluster] # start with prior (assume uniform distribution)
        for attr in range(len(attr_info)): # For every pixel in image
            loc = attr_info[attr][0]
            var = attr_info[attr][1]
            likelihood *= normal.pdf(feature[attr], loc=loc, scale=np.sqrt(var)) # log(a*b) = log(a) + log(b) is a good way to eliminate the zero variance problem.
        return likelihood


    def calculate_gaussian(self, num_attrs, cluster):
        attr_info = []
        for attr in range(num_attrs): # For every pixel in image
            # Mean, standard deivation for pixel in all features
            attr_mean = np.mean([feature[attr] for feature in self.cluster_features[cluster]])
            attr_var = np.var([feature[attr] for feature in self.cluster_features[cluster]]) + self.epsilon # Add epsilon to variances
            attr_info.append((attr_mean, attr_var))
        return attr_info

    def em_steps(self, features):
        # 2.1 Expectation - calculate residual based on posterior, see GMM.
        self.r = np.zeros((len(features), self.num_clusters))
        for i, feature in tqdm(enumerate(features)):
            # Calculate responsibility for every feature
            r_ik = [self.probability(feature, cluster) for cluster in range(self.num_clusters)]
            self.r[i] = r_ik / np.sum(r_ik)

        # 2.2 Maximization
        min_convergence_level = np.inf
        for cluster in tqdm(range(self.num_clusters)):
            r_k = np.sum(self.r, axis=0)[cluster]
            self.cluster_priors[cluster] = r_k / len(features)

            # Update means
            new_means_k = np.sum([self.r[i][cluster] * feature for i, feature in enumerate(features)], axis=0) / r_k
            mean_change = norm(new_means_k - self.cluster_center_means[cluster])
            if mean_change < min_convergence_level: min_convergence_level = mean_change
            self.cluster_center_means[cluster] = new_means_k

            # Update variances
            new_covars_k_1 = np.sum([self.r[i][cluster]*np.matrix(feature).T*feature for i, feature in enumerate(features)], axis=0) / r_k
            new_covars_k_2 = np.matrix(new_means_k).T*new_means_k
            new_covars_k = new_covars_k_1 - new_covars_k_2
            new_vars = np.diagonal(new_covars_k) + self.epsilon

            # Update stored gaussian info for next iteration
            new_attr_info = list(zip(new_means_k, new_vars))
            self.gaussian_info[cluster] = new_attr_info
        
        # Save convergence level, the minimum amount of change occurred in a cluster center between two iterations
        self.convergence_level = min_convergence_level


    def fit(self, features):
        # 1. Initialize random clusters with means and variances
        splits = np.array_split(features, self.num_clusters)
        for cluster in range(self.num_clusters):
            self.cluster_priors[cluster] = 1 / self.num_clusters
            self.cluster_features[cluster] = splits[cluster] # One tenth of a digits dataset
            self.gaussian_info[cluster] = self.calculate_gaussian(features.shape[1], cluster)
            self.cluster_center_means[cluster] = [x[0] for x in self.gaussian_info[cluster]]

        # 2. EM-algorithm            
        self.convergence_level = np.inf
        iteration = 0
        while self.convergence_level > self.threshold:
            self.em_steps(features)
            iteration += 1
            print(f'Iteration {iteration}: convergence_level: {self.convergence_level}')


    def plot_clusters(self, figsize=(10, 10)):
        centers = np.asarray(self.cluster_centers)
        centers = centers.reshape(centers.shape[0], int(centers.shape[1]**(1/2)), int(centers.shape[1]**(1/2)))
        fig = plt.figure(figsize=figsize)
        columns = 3
        rows = 4
        for i in range(0, centers.shape[0]):
            fig.add_subplot(rows, columns, i + 1, title=f'Cluster {i}')
            plt.imshow(centers[i])
        plt.show()


    def transform_labels(self, predicted, mapping):
        labels = [0]*len(predicted)
        for i, x in enumerate(predicted):
            labels[i] = mapping[x]
        return np.asarray(labels)


    def predict(self, features):
        self.cluster_centers = np.zeros((10, 64))
        for cluster in range(self.num_clusters):
            means = [x[0] for x in self.gaussian_info[cluster]]
            self.cluster_centers[cluster] = means
        
        convergence_level = np.inf
        while convergence_level > self.threshold:
            # E-step
            predicted_clusters = [np.argmin([norm(feature - [x[0] for x in self.gaussian_info[cluster]]) for cluster in self.gaussian_info.keys()]) for feature in features]

            # M-step
            new_cluster_centers = [np.mean([feature for feature_ind, feature in enumerate(features) if predicted_clusters[feature_ind] == cluster], axis=0) for cluster in range(self.num_clusters)]

            # Check convergence:
            min_change = np.min([norm(self.cluster_centers[c] - new_cluster_centers[c]) for c in range(self.num_clusters)])
            if min_change < convergence_level: convergence_level = min_change
            self.cluster_centers = new_cluster_centers

        return np.asarray(predicted_clusters)