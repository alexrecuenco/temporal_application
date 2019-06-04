from plotnine import *
from kmeans import KMeans
from test import rand_point
import pandas as pd

if __name__ == "__main__":

    points = [rand_point(2) for _ in range(200)]

    kmeans = KMeans(3).fit(points)

    predictions = kmeans.predict(points)

    table = pd.DataFrame(points)
    table["classification"] = predictions
    print(table)
    centroids = kmeans.centroids

    plot = ggplot(aes(x=0, y=1)) + geom_point(aes(color="factor(classification)"))
    plot.save(filename="test3.png", height=5, width=5, units="in", dpi=1000)
