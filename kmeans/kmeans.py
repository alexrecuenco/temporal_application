# Using typing
from typing import Tuple, Set, List, Dict
from collections import defaultdict, Counter
from random import shuffle


# We assume no libraries outside of the standard library due to the statement in the problem

# The expected returned value from the function is 1) a set of cluster id that each point belongs to, and 2) coordinates of centroids at the end of iteration.

Point = Tuple[float, ...]
Cluster = List[Point]
Centroids = Cluster


# We assume eucledian kernel
def kernel(p1: Point, p2: Point) -> float:
    return sum((x1 - x2) ** 2 for (x1, x2) in zip(p1, p2))


def get_center(points: Cluster) -> Point:
    points_it = iter(points)
    sum_points = list(next(points_it))
    if sum_points is None:
        raise TypeError("Empty list given to get_center")

    npoints = len(points)
    # TODO: Use reduce/functools to put this part in C-code
    # TODO: Do a double pass for extra precission, right now this is not numerically sound
    for point in points_it:
        for coordinate, value in enumerate(point):
            sum_points[coordinate] += value

    return tuple(val / npoints for val in sum_points)


def classify_randomly(k: int, points: Cluster) -> Dict[int, Cluster]:

    all_idxs = list(range(len(points)))
    shuffle(all_idxs)

    curr_class = 0
    classification: Dict[int, Cluster] = defaultdict(list)

    for idx in all_idxs:
        classification[curr_class].append(points[idx])
        curr_class = (curr_class + 1) % k

    return classification


def closest_centroid_idx(centroids: Centroids, point: Point) -> int:
    closest_idx_centroid = min(
        enumerate(centroids), key=lambda idx_centroid: kernel(idx_centroid[1], point)
    )
    return closest_idx_centroid[0]


def classify(centroids: Centroids, points: Cluster) -> Dict[int, Cluster]:
    """classify We assume the centroids are ordered
    
    Arguments:
        centroids {Centroids} -- [description]
        points {Cluster} -- [description]
    
    Returns:
        Dict[int, Cluster] -- [description]
    """

    classification: Dict[int, Cluster] = defaultdict(list)

    for point in points:
        idx = closest_centroid_idx(centroids, point=point)
        classification[idx].append(point)

    return classification


def is_same_grouping(
    k: int, classification1: Dict[int, Cluster], classification2: Dict[int, Cluster]
) -> bool:
    """Returns whether the two k-classifications are identically classified 
        
    Arguments:
        classification1 {Dict[int, Cluster]} -- [description]
        classification2 {Dict[int, Cluster]} -- [description]
    
    Returns:
        bool -- [description]
    """

    for cls_idx in range(k):
        item1 = classification1[cls_idx]
        item2 = classification2[cls_idx]
        if Counter(item1) != Counter(item2):
            return False

    return True


from operator import itemgetter


def get_centroids(classification: Dict[int, Cluster]) -> Centroids:
    idx_and_centroids = [
        (cls_idx, get_center(group)) for cls_idx, group in classification.items()
    ]
    idx_and_centroids.sort(key=itemgetter(0))
    return list(centroid for _, centroid in idx_and_centroids)


class KMeansModel:
    def __init__(self, centroids: Centroids):
        self.centroids = centroids

    def predict(self, X: Cluster) -> List[int]:
        if self.centroids is None:
            raise ValueError("You first need to fit some data to this model")
        return [closest_centroid_idx(self.centroids, point) for point in X]


class KMeans:
    # Let's imitate a bit of the sklearn standard
    def __init__(self, n_cluster: int, max_iter=500):
        self.n_cluster = n_cluster
        self.max_iter = max_iter
        # We could reimplement the whole kmean algorithm... but let's just follow a tiny bit only the structure

    def fit(self, X):
        groups = classify_randomly(self.n_cluster, X)
        centroids = get_centroids(groups)

        groups_new = classify(centroids, X)
        n_tries_left = self.max_iter

        while (
            not is_same_grouping(self.n_cluster, groups, groups_new)
            and n_tries_left > 0
        ):
            n_tries_left -= 1
            groups = groups_new
            centroids = get_centroids(groups)
            groups_new = classify(centroids, X)

        if n_tries_left == 0:
            print("WARNING: The method didn't converge")

        return KMeansModel(centroids)

    def fit_predict(self, X):
        kmeans = self.fit(X)
        return (kmeans.centroids, kmeans.predict(X))

