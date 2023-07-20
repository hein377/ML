# DATASET SETUP
from sklearn.datasets import make_blobs
from sklearn.datasets import make_circles

# P1 DATASET
def make_circle_dataset(): return make_circles(n_samples=500, noise=0.05, factor=0.2)
x1, y1 = make_circle_dataset()

def make_planets_dataset(): return make_blobs(n_samples=[500, 500, 20, 20], centers=[(-10, 5), (10, 5), (0, 10), (0, 0)], cluster_std=[1.3, 1.3, .5, .5])
x2, y2 = make_planets_dataset()

# P2 DATASET
def make_planets_dataset2(): return make_blobs(n_samples=[200, 4000, 200], centers=[(-9, 0), (0, 0), (9, 0)], cluster_std=[1.25, 2.5, 1.25])
x3, y3 = make_planets_dataset2()

# KMEANS
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
def k_means(x, y, k_val, initializing_technique):
    labels = KMeans(k_val, init=initializing_technique).fit_predict(x)
    plt.scatter(x[:,0], x[:,1], c=labels, s=50, cmap='viridis')
    plt.show()

# TESTING K
#initializing_technique - 'k-means++' or 'random' or list of tuples for centroids
k_means(x1, y1, 3, [(-.75,-.5), (0,1), (.75,-.5)])

for k in range(2,6):
    k_means(x1, y1, k, 'random')
    k_means(x2, y2, k, 'random')
    k_means(x3, y3, k, 'random')

for k in range(2,6):
    k_means(x1, y1, k, 'k-means++')
    k_means(x2, y2, k, 'k-means++')
    k_means(x3, y3, k, 'k-means++')