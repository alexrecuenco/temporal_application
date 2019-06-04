import plotnine
from kmeans import KMeans
from test import rand_point
import pandas as pd

if __name__ == "__main__":

    points = [rand_point(2) for _ in range(200)]

    kmeans = KMeans(3).fit(points)

    predictions = kmeans.predict(points)

    pd.DataFrame(points)
    pd["classification"] = predictions
    centroids = kmeans.centroids

    plotnine
