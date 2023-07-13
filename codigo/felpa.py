import numpy as np
import matplotlib.pyplot as plt
import csv
import random
plt.style.use('tp03.mplstyle')
# from sklearn.cluster import KMeans

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

r = 3
T = T[:, :r]
added = [False for i in range(0, len(T))]

rand1= random.randrange(0, 2000)
seed = T[rand1]

def dist(a, b):
    new = []
    dist = 0
    for i in range(len(a)):
        new.append(a[i]-b[i])
    for i in new:
        dist += i**2
    return dist**0.5

print(seed)
# for i in range(0, len(T)):
#     print(dist(seed, T[i]))

cluster1 = []
cluster2 = []

cluster1.append(seed)
added[rand1] = True

def cluster():
    # print(len(cluster1)) 
    done = True
    for i in cluster1:
        for j in range(0, len(T)):
            if added[j] == False and dist(T[j], i) <= 0.25:
                done = False
                cluster1.append(T[j])
                added[j] = True
        if not done:
            cluster()
        if done:
            return

cluster()
for i in T:
    it_belongs = True
    for j in cluster1:
        if list(i) == list(j):
            it_belongs = False
    if it_belongs:
        cluster2.append(i)

c1x = [cluster1[i][0] for i in range(0, len(cluster1))]
c1y = [cluster1[j][1] for j in range(0, len(cluster1))]
c2x = [cluster2[k][0] for k in range(0, len(cluster2))]
c2y = [cluster2[l][1] for l in range(0, len(cluster2))]


x_coords_c1 = np.array(c1x)
y_coords_c1 = np.array(c1y)
x_coords_c2 = np.array(c2x)
y_coords_c2 = np.array(c2y)


print(len(cluster2))
plt.scatter(x_coords_c2, y_coords_c2, s=18, c= "#9e0142", edgecolors='black',linewidths=0.5)
plt.scatter(x_coords_c1, y_coords_c1, s=18, c="#5e4fa2", edgecolors='black',linewidths=0.5)
plt.xlabel("Posición en x")
plt.ylabel("Posición en y")
# plt.title("Visualización de la clasificación en clusters")
plt.savefig("plotdeclustersfelpa.pdf", format='pdf', dpi=1200)
plt.show()