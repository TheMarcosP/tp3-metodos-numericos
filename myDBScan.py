import numpy as np
import csv
import random
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

import numpy as np

def dbscan(X, eps, min_samples):
    labels = np.zeros(len(X), dtype=int)  # Cluster labels initialization
    cluster_id = 0  # Cluster ID initialization

    dist_matrix = compute_distance_matrix(X)  # Compute distance matrix

    for i in range(len(X)):
        if labels[i] != 0:  # Skip points already assigned to a cluster or marked as noise
            continue

        neighbors = np.where(dist_matrix[i] < eps)[0]  # Find neighbors within the epsilon distance

        if len(neighbors) < min_samples:  # Assign noise label to points with insufficient neighbors
            labels[i] = -1
        else:
            cluster_id += 1
            expand_cluster(X, labels, i, neighbors, cluster_id, eps, min_samples, dist_matrix)

    return labels

def compute_distance_matrix(X):
    n = len(X)
    dist_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            dist_matrix[i, j] = np.linalg.norm(X[i] - X[j])
            dist_matrix[j, i] = dist_matrix[i, j]
    return dist_matrix

def expand_cluster(X, labels, point_idx, neighbors, cluster_id, eps, min_samples, dist_matrix):
    labels[point_idx] = cluster_id
    i = 0
    while i < len(neighbors):
        neighbor_idx = neighbors[i]
        if labels[neighbor_idx] == -1:  # Change noise label to border point
            labels[neighbor_idx] = cluster_id
        elif labels[neighbor_idx] == 0:  # Expand cluster to unvisited points
            labels[neighbor_idx] = cluster_id
            new_neighbors = np.where(dist_matrix[neighbor_idx] < eps)[0]  # Use distance matrix to find neighbors
            if len(new_neighbors) >= min_samples:
                neighbors = np.concatenate((neighbors, new_neighbors))
        i += 1

def plot_dbsan(X, labels):

    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)

    fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d') # 3d
    ax = fig.add_subplot(111) # 2d

    # Plot the clusters
    unique_labels = set(labels)
    colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]
    for k, col in zip(unique_labels, colors):
        if k == -1:
            col = [0, 0, 0, 1]  # Black color for noise points

        class_member_mask = (labels == k)
        xyz = X[class_member_mask]
        # ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], c=tuple(col), edgecolors='k', s=50) #3d
        ax.scatter(xyz[:, 0], xyz[:, 1], c=tuple(col), edgecolors='k', s=50) #2d

    ax.set_title('Estimated number of clusters: %d' % n_clusters_)
    print("Estimated number of noise points: %d" % n_noise_)

    # Calculate the centroids of each cluster
    centroids = []
    for i in range(1, n_clusters_+1):
        cluster = []
        for j in range(len(labels)):
            if labels[j] == i:
                cluster.append(X[j])
        cluster = np.array(cluster)
        centroid = np.mean(cluster, axis=0)
        centroids.append(centroid)

    centroids = np.array(centroids)

    # Plot the centroids
    for i in range(len(centroids)):
        print("Centroid of cluster", i+1, "is", centroids[i])
        # ax.scatter(centroids[i][0], centroids[i][1], centroids[i][2], c='red', marker='X',s=200, zorder=10) # 3d
        ax.scatter(centroids[i][0], centroids[i][1], c='red', marker='X', s=200) # 2d

    plt.show()

    # Count the number of points in each cluster
    for i in range(1, n_clusters_+1):
        count = 0
        for j in range(len(labels)):
            if labels[j] == i:
                count += 1
        print("Cluster", i, "has", count, "points")

def find_best_params(X, epsilons, min_samples):
    # Encontrar el mejor epsilon y min_samples
    # https://www.youtube.com/watch?v=VO_uzCU_nKw&ab_channel=GregHogg
    # basicamente prueba todas las combinaciones de epsilon y le asigna un score a cada combinacion usando el silhouette score
    from sklearn.metrics import silhouette_score
    best_score = -1
    best_params = (0, 0)
    for eps in epsilons:
        for min_sample in min_samples:
            labels = dbscan(X, eps, min_sample)
            labels_set = set(labels)
            num_clusters = len(labels_set)

            if -1 in labels_set:
                num_clusters -= 1
            if (num_clusters < 2) or (num_clusters > 50):
                print(f"Combination {eps}, {min_sample} has {num_clusters} clusters. Moving on")
                continue

            score = silhouette_score(X, labels)
            print("score:",score,"with params:", (eps, min_sample))

            if score > best_score:
                best_score = score
                best_params = (eps, min_sample)
    return best_params


def main():
    
    file = 'dataset_clusters.csv'

    array = []

    with open(file, 'r') as data:
        reader = csv.reader(data)
        for i in reader:
            array.append(i)

    A = np.array(array, dtype=np.float64)

    C = (np.transpose(A)) @ A  # matriz de covarianza
    eigenvalues, eigenvectors = np.linalg.eig(C)
    V = np.array(eigenvectors)
    TT= A @ V

    r = 2  # Use 3d / 2d
    TT = TT[:, :r]



    # epsilons = np.linspace(0.01, 1, num=15)
    # min_samples = np.arange(2, 20, step=3)
    # # epsilons = np.array([0.1, 0.7878, 1])
    # # min_samples = np.array([2, 8, 11, 14])

    # best_eps, best_min_sample = find_best_params(TT, epsilons, min_samples)
    # print("Best parameters:", best_eps, best_min_sample)


    # uso los mejores para plotear
    labels = dbscan(TT, 0.7878571428571429, 14)
    plot_dbsan(TT, labels)









# import itertools

# combinations = list(itertools.product(epsilons, min_samples))

# N = len(combinations)

# def get_scores_and_labels(combinations, X):
#   scores = []
#   all_labels_list = []

#   for i, (eps, num_samples) in enumerate(combinations):
#     labels = dbscan(X, eps, num_samples)
  
#     labels_set = set(labels)
#     num_clusters = len(labels_set)
#     if -1 in labels_set:
#       num_clusters -= 1
    
#     if (num_clusters < 2) or (num_clusters > 50):
#       scores.append(-10)
#       all_labels_list.append('bad')
#       c = (eps, num_samples)
#       print(f"Combination {c} on iteration {i+1} of {N} has {num_clusters} clusters. Moving on")
#       continue
    
#     scores.append(ss(X, labels))
#     all_labels_list.append(labels)
#     print(f"Index: {i}, Score: {scores[-1]}, Labels: {all_labels_list[-1]}, NumClusters: {num_clusters}")

#   best_index = np.argmax(scores)
#   best_parameters = combinations[best_index]
#   best_labels = all_labels_list[best_index]
#   best_score = scores[best_index]

#   return {'best_epsilon': best_parameters[0],
#           'best_min_samples': best_parameters[1], 
#           'best_labels': best_labels,
#           'best_score': best_score}

# best_dict = get_scores_and_labels(combinations, T)


# # resultado

# # Combination (0.01, 2) on iteration 1 of 90 has 55 clusters. Moving on
# # Combination (0.01, 5) on iteration 2 of 90 has 0 clusters. Moving on
# # Combination (0.01, 8) on iteration 3 of 90 has 0 clusters. Moving on
# # Combination (0.01, 11) on iteration 4 of 90 has 0 clusters. Moving on
# # Combination (0.01, 14) on iteration 5 of 90 has 0 clusters. Moving on
# # Combination (0.01, 17) on iteration 6 of 90 has 0 clusters. Moving on
# # Combination (0.08071428571428571, 2) on iteration 7 of 90 has 177 clusters. Moving on
# # Combination (0.08071428571428571, 5) on iteration 8 of 90 has 67 clusters. Moving on
# # Index: 8, Score: -0.42160021020537436, Labels: [-1 -1 -1 ... -1 -1 18], NumClusters: 27
# # Index: 9, Score: -0.46230348897165474, Labels: [-1 -1 -1 ... -1 -1 13], NumClusters: 14
# # Index: 10, Score: -0.49170646991246475, Labels: [-1 -1 -1 ... -1 -1 -1], NumClusters: 6
# # Index: 11, Score: -0.5126037005063129, Labels: [-1 -1 -1 ... -1 -1 -1], NumClusters: 4
# # Index: 12, Score: -0.26907256053521633, Labels: [ 1  1  1 ... 16 16 16], NumClusters: 27
# # Index: 13, Score: -0.17942186144230268, Labels: [1 1 1 ... 7 7 7], NumClusters: 12
# # Index: 14, Score: -0.11159637342731683, Labels: [1 1 1 ... 6 6 6], NumClusters: 10
# # Index: 15, Score: 0.009681139244675634, Labels: [ 1  1  2 ...  4 -1  4], NumClusters: 6
# # Index: 16, Score: -0.03793491008066114, Labels: [-1  3 -1 ...  6 -1  6], NumClusters: 8
# # Index: 17, Score: -0.07744701476384185, Labels: [-1  3 -1 ...  5 -1  5], NumClusters: 6
# # Index: 18, Score: -0.03603994278086199, Labels: [1 1 1 ... 3 3 3], NumClusters: 8
# # Index: 19, Score: 0.19358771333655966, Labels: [1 1 1 ... 3 3 3], NumClusters: 3
# # Index: 20, Score: 0.21669898710399452, Labels: [1 1 1 ... 3 3 3], NumClusters: 3
# # Index: 21, Score: 0.24484856263932486, Labels: [1 1 1 ... 2 2 2], NumClusters: 3
# # Index: 22, Score: 0.3830725659753115, Labels: [1 1 1 ... 2 2 2], NumClusters: 2
# # Index: 23, Score: 0.1292938108214186, Labels: [1 1 1 ... 3 3 3], NumClusters: 4
# # Index: 24, Score: 0.15838571588566344, Labels: [1 1 1 ... 2 2 2], NumClusters: 4
# # Index: 25, Score: 0.36694368708495667, Labels: [1 1 1 ... 2 2 2], NumClusters: 2
# # Index: 26, Score: 0.3750943260371605, Labels: [1 1 1 ... 2 2 2], NumClusters: 2
# # Index: 27, Score: 0.3722607104002498, Labels: [1 1 1 ... 2 2 2], NumClusters: 2
# # Index: 28, Score: 0.36481848350682183, Labels: [1 1 1 ... 2 2 2], NumClusters: 2
# # Index: 29, Score: 0.3709311820843232, Labels: [1 1 1 ... 2 2 2], NumClusters: 2
# # Index: 30, Score: 0.19319372602356533, Labels: [1 1 1 ... 2 2 2], NumClusters: 3
# # Index: 31, Score: 0.3915163186208895, Labels: [1 1 1 ... 2 2 2], NumClusters: 2
# # Index: 32, Score: 0.36904160228681804, Labels: [1 1 1 ... 2 2 2], NumClusters: 2
# # Index: 33, Score: 0.3455514577453067, Labels: [1 1 1 ... 2 2 2], NumClusters: 2
# # Index: 34, Score: 0.3519015063549981, Labels: [1 1 1 ... 2 2 2], NumClusters: 2
# # Index: 35, Score: 0.36724461244238193, Labels: [1 1 1 ... 2 2 2], NumClusters: 2
# # Index: 36, Score: 0.18740243647932903, Labels: [1 1 1 ... 2 2 2], NumClusters: 3
# # Index: 37, Score: 0.38866128044716763, Labels: [1 1 1 ... 2 2 2], NumClusters: 2
# # Index: 38, Score: 0.3612307169991827, Labels: [1 1 1 ... 2 2 2], NumClusters: 2
# # Index: 39, Score: 0.36284605753809374, Labels: [1 1 1 ... 2 2 2], NumClusters: 2
# # Index: 40, Score: 0.35581521019517387, Labels: [1 1 1 ... 2 2 2], NumClusters: 2
# # Index: 41, Score: 0.3638908645357262, Labels: [1 1 1 ... 2 2 2], NumClusters: 2
# # Combination (0.505, 2) on iteration 43 of 90 has 1 clusters. Moving on
# # Index: 43, Score: 0.29441678336836774, Labels: [1 1 1 ... 2 2 2], NumClusters: 2
# # Index: 44, Score: 0.3623767103621539, Labels: [1 1 1 ... 2 2 2], NumClusters: 2
# # Index: 45, Score: 0.3623767103621539, Labels: [1 1 1 ... 2 2 2], NumClusters: 2
# # Index: 46, Score: 0.34948590595262546, Labels: [1 1 1 ... 2 2 2], NumClusters: 2
# # Index: 47, Score: 0.3352173677834972, Labels: [1 1 1 ... 2 2 2], NumClusters: 2
# # Combination (0.5757142857142857, 2) on iteration 49 of 90 has 1 clusters. Moving on
# # Combination (0.5757142857142857, 5) on iteration 50 of 90 has 1 clusters. Moving on
# # Index: 50, Score: 0.3437174702381933, Labels: [1 1 1 ... 2 2 2], NumClusters: 2
# # Combination (0.6464285714285715, 2) on iteration 55 of 90 has 1 clusters. Moving on
# # Combination (0.6464285714285715, 5) on iteration 56 of 90 has 1 clusters. Moving on
# # Index: 56, Score: 0.5387639486753631, Labels: [1 1 1 ... 2 2 2], NumClusters: 2
# # Index: 57, Score: 0.5387639486753631, Labels: [1 1 1 ... 2 2 2], NumClusters: 2
# # Index: 58, Score: 0.5387639486753631, Labels: [1 1 1 ... 2 2 2], NumClusters: 2
# # Index: 59, Score: 0.5387639486753631, Labels: [1 1 1 ... 2 2 2], NumClusters: 2
# # Combination (0.7171428571428572, 2) on iteration 61 of 90 has 1 clusters. Moving on
# # Combination (0.7171428571428572, 5) on iteration 62 of 90 has 1 clusters. Moving on
# # Index: 63, Score: 0.5387841208548171, Labels: [1 1 1 ... 2 2 2], NumClusters: 2
# # Index: 64, Score: 0.5390817007018457, Labels: [1 1 1 ... 2 2 2], NumClusters: 2
# # Index: 65, Score: 0.5390817007018457, Labels: [1 1 1 ... 2 2 2], NumClusters: 2
# # Combination (0.7878571428571429, 2) on iteration 67 of 90 has 1 clusters. Moving on
# # Combination (0.7878571428571429, 5) on iteration 68 of 90 has 1 clusters. Moving on
# # Combination (0.7878571428571429, 8) on iteration 69 of 90 has 1 clusters. Moving on
# # Combination (0.7878571428571429, 11) on iteration 70 of 90 has 1 clusters. Moving on
# # Index: 70, Score: 0.5392392562002316, Labels: [1 1 1 ... 2 2 2], NumClusters: 2            Este es el mejor score  con epsilon y min samples = (0.7878571428571429, 14)
# # Index: 71, Score: 0.5390817007018457, Labels: [1 1 1 ... 2 2 2], NumClusters: 2
# # Combination (0.8585714285714285, 2) on iteration 73 of 90 has 1 clusters. Moving on
# # Combination (0.8585714285714285, 17) on iteration 78 of 90 has 1 clusters. Moving on
# # Combination (0.9292857142857143, 2) on iteration 79 of 90 has 1 clusters. Moving on
# # Combination (0.9292857142857143, 5) on iteration 80 of 90 has 1 clusters. Moving on
# # Combination (0.9292857142857143, 8) on iteration 81 of 90 has 1 clusters. Moving on
# # Combination (0.9292857142857143, 11) on iteration 82 of 90 has 1 clusters. Moving on
# # Combination (0.9292857142857143, 14) on iteration 83 of 90 has 1 clusters. Moving on
# # Combination (0.9292857142857143, 17) on iteration 84 of 90 has 1 clusters. Moving on
# # Combination (1.0, 2) on iteration 85 of 90 has 1 clusters. Moving on
# # Combination (1.0, 5) on iteration 86 of 90 has 1 clusters. Moving on
# # Combination (1.0, 8) on iteration 87 of 90 has 1 clusters. Moving on
# # Combination (1.0, 11) on iteration 88 of 90 has 1 clusters. Moving on
# # Combination (1.0, 14) on iteration 89 of 90 has 1 clusters. Moving on
# # Combination (1.0, 17) on iteration 90 of 90 has 1 clusters. Moving on



if __name__ == "__main__":
    main()