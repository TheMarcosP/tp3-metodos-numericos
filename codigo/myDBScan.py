import numpy as np
import csv
import random
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.lines import Line2D

import numpy as np


def dbscan(X, eps, min_neighbours):
    labels = np.zeros(len(X), dtype=int)  
    cluster_id = 0  

    dist_matrix = compute_distance_matrix(X)  

    for i in range(len(X)):
        if labels[i] != 0:  # salteo los puntos que ya pertenecen a un cluster o estan marcados como ruido
            continue

        neighbors = np.where(dist_matrix[i] < eps)[0]  # encuentro los vecinos del punto i

        if len(neighbors) < min_neighbours: 
            labels[i] = -1
        else:
            cluster_id += 1
            expand_cluster(X, labels, i, neighbors, cluster_id, eps, min_neighbours, dist_matrix)

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
        if labels[neighbor_idx] == -1:  
            labels[neighbor_idx] = cluster_id
        elif labels[neighbor_idx] == 0:  
            labels[neighbor_idx] = cluster_id
            new_neighbors = np.where(dist_matrix[neighbor_idx] < eps)[0]  
            if len(new_neighbors) >= min_samples:
                neighbors = np.concatenate((neighbors, new_neighbors))
        i += 1


def plot_dbscan(X, labels, filename):

    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)

    fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d') # 3d
    ax = fig.add_subplot(111) # 2d

    # Plot the clusters
    unique_labels = set(labels)
    colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]
    legend_elements = []

    for k, col in zip(unique_labels, colors):
        if k == -1:
            col = [0, 0, 0, 1]  # Black color for noise points

        class_member_mask = (labels == k)
        xyz = X[class_member_mask]
        # scatter = ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], c=tuple(col), edgecolors='k', s=50) #3d
        scatter = ax.scatter(xyz[:, 0], xyz[:, 1], c=tuple(col), edgecolors='k', s=50) #2d
        legend_elements.append(Line2D([0], [0], marker='o', color='w', label= f'Cluster {k}' if not k==-1 else "outliers", markerfacecolor=scatter.get_facecolor()[0], markersize=5))

    print('Estimated number of clusters: %d' % n_clusters_)
    print("Estimated number of noise points: %d" % n_noise_)

    # Calculate the centroids of each cluster
    centroids = compute_centroids(X, labels)

    # Plot the centroids
    for i in range(len(centroids)):
        print("Centroid of cluster", i+1, "is", centroids[i])
        # ax.scatter(centroids[i][0], centroids[i][1], centroids[i][2], c='red', marker='X', s=200, zorder=10) # 3d
        ax.scatter(centroids[i][0], centroids[i][1], c='red', marker='X', s=200) # 2d
        # add legend for centroids
        legend_elements.append(Line2D([0], [0], marker='X', color='w', label=f'Centroid {i+1}', markerfacecolor='red', markersize=10))

    # Add legend
    ax.legend(handles=legend_elements, loc='upper right')

    plt.savefig(filename, format='pdf', dpi=1200)

def count_points_in_clusters(labels):
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
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
            print("score:",score,"with params:", (eps, min_sample), "number of clusters: ", num_clusters)

            if score > best_score:
                best_score = score
                best_params = (eps, min_sample)
    return best_params


def compute_centroids(X, labels):

    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
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
    return centroids


def cluster_points(X, centroids):
    """given the data points and the centroids, assign each point to the closest centroid
    """
    labels = []
    for i in range(len(X)):
        distances = []
        for j in range(len(centroids)):
            distances.append(np.linalg.norm(X[i] - centroids[j]))
        labels.append(np.argmin(distances)+1)
    return labels

def calculate_avg_distance(X):
    """aproximate the avg distance between points"""

    distances = []
    for i in range(len(X)):
        for j in range(i+1, len(X)):
            distances.append(np.linalg.norm(X[i] - X[j]))
    return np.mean(distances)



def main():
    
    file = 'dataset_clusters.csv'

    array = []

    with open(file, 'r') as data:
        reader = csv.reader(data)
        for i in reader:
            array.append(i)

    A = np.array(array, dtype=np.float64)

    epsilons = np.linspace(3, 4, num=5)
    print("epsilons: ", epsilons)
    min_samples = np.arange(10, 60, step=15)
    print("min_samples: ", min_samples)

    values = ((3,0.7714285714, 2),(4, 1.3823529411, 33),(6, 1.6, 22), (8, 2.325, 40))
    for i in range(4):
        d, best_eps, best_min_sample = values[i]
        # A shape is (2000,106)
       
        #calculo el promedio de cada fila
        A_mean = np.mean(A, axis=0)

        # centro los datos
        X_centered = A - A_mean

        # hago SVD
        U, S, Vt = np.linalg.svd(X_centered)

        top_d_singular_vectors = Vt[:d].T

        # proyectamos los datos en el nuevo espacio
        a = np.dot(X_centered, top_d_singular_vectors)

        epsilons = np.array([0.685714])
        min_samples = np.array([14])

        # best_eps, best_min_sample = find_best_params(a, epsilons, min_samples)
        # print("Best parameters:", best_eps, best_min_sample)

        from sklearn.metrics import silhouette_score
        # uso los mejores para plotear
        labels = dbscan(a,  best_eps, best_min_sample)
        print(f"para {d} dimensiones, con epsilon y min samples = {best_eps, best_min_sample} ")
        count_points_in_clusters(labels)
        # plot_dbscan(a, labels, "dbscan2d.pdf")
        print(f"Silhouette score: {silhouette_score(a, labels)}")

        # use the centroids to find the clusters
        # centroids = compute_centroids(a, labels)
        # labels = cluster_points(a, centroids)
        # count_points_in_clusters(labels)
        # plot_dbscan(a, labels, f"CentroidsClustering{d}.pdf")

        # avg_distance = calculate_avg_distance(a)
        # print("avg distance:", avg_distance)



#para 4 dimensiones
# score: 0.3479028582129058 with params: (1.3823529411764706, 33) number of clusters:  2

#con la nueva forma de pca

# para 10 dimensiones, con epsilon y min samples = (2.7368421052631575, 35) 
# Cluster 1 has 707 points
# Cluster 2 has 675 points


# viejo
# dimensiones, epsilon, min neighbors, cluster 1, cluster 2
# 2, 0.685714, 14, 1002, 998
# 3, 0.955555, 14, 991, 991
# 4, 1.366666, 31,976, 961
# 5, 1.542105, 31, 918, 893
# 6, 1.25, 20, 975, 967
# 7, 1.357142, 10, 944, 922
# 8, 1.785714, 35, 850, 812
# 9, 1.785714, 10, 813, 795
# 10, 1.785714, 10, 813, 795
# 11, 1.785714, 10, 813, 795
# 12, 2.0, 15, 742, 744
# 13, 2.571428, 44, 728, 722

# r, eps, min_samples, cluster1, cluster2
# 2,
# 3,
# 4, 0.7878571428571429, 20, 1000, 1000
# 5, 1.0357142857142856, 20, 997, 983
# 6, 1.25, 20, 975, 967
# 7, 1.3571428571428572, 10, 944, 922
# 8, 1.7857142857142856, 35, 850, 812
# 9, 1.7857142857142856, 10, 813, 795
# 10, 1.7857142857142856, 10, 813, 795
# 11, 1.7857142857142856, 10, 813, 795
# 12, 2.0, 15, 742, 744
# 13, 2.571428, 44, 728, 722


if __name__ == "__main__":
    main()