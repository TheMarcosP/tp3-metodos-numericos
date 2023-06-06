from sklearn.cluster import DBSCAN
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
import numpy as np


import numpy as np
import matplotlib.pyplot as plt
import csv
import random

file = 'dataset_clusters.csv'

array = []

with open(file, 'r') as data:
    reader = csv.reader(data)
    for i in reader:
        array.append(i)

A = np.array(array, dtype=np.float64)

C = (np.transpose(A)) @ A # matriz de covarianza
eigenvalues, eigenvectors = np.linalg.eig(C)
V = np.array(eigenvectors)
T = A @ V

r = 2
X = T[:, :r]



# Create an instance of the DBSCAN algorithm
dbscan = DBSCAN(eps=0.4, min_samples=5)

# Fit the data to the DBSCAN algorithm
dbscan.fit(X)

# Obtain the predicted cluster labels
labels = dbscan.labels_

# Number of clusters in labels, ignoring noise if present.
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
n_noise_ = list(labels).count(-1)

# Plot the clusters
unique_labels = set(labels)
colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]
for k, col in zip(unique_labels, colors):
    if k == -1:
        col = [0, 0, 0, 1]  # Black color for noise points

    class_member_mask = (labels == k)

    xy = X[class_member_mask]
    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col), markeredgecolor='k', markersize=6)

plt.title('Estimated number of clusters: %d' % n_clusters_)
plt.show()
