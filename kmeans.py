import random
import math

'''
Pure python implementation of Kmeans algorithm
'''

def euclidian_distance(vec1, vec2):
    '''
    returns the euclidian distance b/w
    two vectors
    vec1, vec2 -> should be list of numbers of
                  same length
    '''
    result = 0

    for i in range(len(vec1)):
        result += (vec1[i] - vec2[i])**2

    result = math.sqrt(result)
    return result


def add_vectors(vec1, vec2):
    '''
    returns a new vector resulting
    from addition of vec1 and vec2
    vec1, vec2 -> should be list of numbers of
                  same length
    '''
    result = []

    for i in range(len(vec1)):
        result.append(vec1[i] + vec2[i])

    return result


def divide_vectors(vec, num):
    '''
    Returns a new vector where every
    element is divided by the num
    '''
    result = []
    for x in vec:
        result.append(x / num)

    return result


class KMeans:

    def __init__(self, n_clusters=8, max_iter=300, tol=0.0001):
        '''
        n_clusters: The number of clusters to form as well as the number of centroids to generate.

        max_iter: Maximum number of iterations of the k-means algorithm for a single run.

        tol: Relative tolerance with regards to inertia to declare convergence
        '''
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol

        # dimention of input vectors
        self.dim_ = None
        # List of cluster centers
        self.cluster_centers_ = []
        # Cluster label of each point
        self.labels_ = []
        # Sum of squared distances of samples to their closest cluster center.
        self.inertia_ = None

    def fit(self, X):
        '''
        Compute k-means clustering.
        '''
        n, self.dim_ = len(X), len(X[0])

        # Randomly initialize clusters centers
        # by randomly selecting points from input dataset
        for i in range(self.n_clusters):
            index = random.randint(0, n - 1)
            self.cluster_centers_.append(X[index])

        # initialize labels
        self.update_labels_(X)

        # initialize inertia
        self.inertia_ = self.compute_inertia_(X)

        for i in range(self.max_iter):

            # Save prev inertia
            prev_inertia = self.inertia_

            # Update cluster label of each datapoint
            self.update_labels_(X)

            # Update the cluster centers based on new labels
            self.update_centroids_(X)

            # Compute current inertia
            self.inertia_ = self.compute_inertia_(X)

            # if change in inertia is less than tolerance value then terminate
            if abs(prev_inertia - self.inertia_) < self.tol:
                break

        return self

    def compute_inertia_(self, X):
        '''
        Compute sum of squared distances of samples
        to their closest cluster center.
        '''
        inertia = 0

        for i in range(len(X)):
            label = self.labels_[i]
            # add the squared euclidian distance
            inertia += euclidian_distance(
                self.cluster_centers_[label], X[i]) ** 2

        return inertia

    def update_labels_(self, X):
        '''
        Updates the class label of each point in X
        '''
        self.labels_ = [self.get_label_(point) for point in X]

    def get_label_(self, point):
        '''
        returns the cluster label of the
        datapoint
        '''

        label = None
        min_distance = math.inf

        # Compute distance from each cluster
        for i, center in enumerate(self.cluster_centers_):
            distance = euclidian_distance(point, center)

            if distance < min_distance:
                label = i
                min_distance = distance

        return label

    def update_centroids_(self, X):
        '''
        Compute new centroids based on labels
        of each datapoints
        '''

        # 2D list of shape same as self.n_clusters
        # to store the sum of all the datapoints of that cluster
        new_centers = [[0] * self.dim_] * self.n_clusters

        # Store number of points that belong to each cluster
        n_points = [0] * self.n_clusters

        for i in range(len(X)):
            label = self.labels_[i]
            new_centers[label] = add_vectors(new_centers[label], X[i])
            n_points[label] += 1

        for i in range(self.n_clusters):
            try:
                # divide the sum of datapoints by number of datapoints
                # to get their mean
                self.cluster_centers_[i] = divide_vectors(
                    new_centers[i], n_points[i])
            except ZeroDivisionError:
                # If no points are assigned to this cluster
                # then keep the cluster point same
                pass


class KMeansRobust:
    '''
    Runs the KMeans multiple times and with different randomly initialized centroids.
    The final results will be the best output of n_init consecutive runs in terms of inertia
    '''

    def __init__(self, n_clusters=8, n_init=10, max_iter=300, tol=0.0001):
        '''
        n_init: Number of time the k-means algorithm will be run

        Rest of the parameters are same as that of KMeans
        '''
        self.n_clusters = n_clusters
        self.n_init = n_init
        self.max_iter = max_iter
        self.tol = tol

        self.dim_ = None
        self.cluster_centers_ = []
        self.labels_ = []
        self.inertia_ = None
        self.best_kmeans = None

    def fit(self, X):
        '''
        Compute k-means clustering.
        '''
        # initialize best inertia with inf
        best_inertia = math.inf

        # Compute KMeans n_init times
        # and select the best
        for i in range(self.n_init):
            kmeans = KMeans(n_clusters=self.n_clusters, max_iter=self.max_iter,
                            tol=self.tol)
            kmeans.fit(X)

            if kmeans.inertia_ < best_inertia:
                best_inertia = kmeans.inertia_
                self.best_kmeans = kmeans

        # Save the attributes of the best kmeans
        self.dim_ = self.best_kmeans.dim_
        self.cluster_centers_ = self.best_kmeans.cluster_centers_
        self.labels_ = self.best_kmeans.labels_
        self.inertia_ = self.best_kmeans.inertia_

        return self
