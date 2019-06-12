from plotnine import *
from kmeans import KMeans
from test import rand_point
import pandas as pd

N_CLUSTER = 6
N_POINTS = 1000
N_DIMENS = 2


if __name__ == "__main__":
    points = [rand_point(N_DIMENS) for _ in range(N_POINTS)]
    kmeans = KMeans(N_CLUSTER).fit(points)
    predictions = kmeans.predict(points)
    table = pd.DataFrame(points)
    table["tag"] = predictions
    table.rename(columns={0: "x", 1: "y"}, inplace=True)
    centroids = kmeans.centroids
    plot = ggplot(aes(x="x", y="y"), table) + geom_point(aes(color="factor(tag)"))
    plot.save(filename="plot.png", height=5, width=5, units="in", dpi=300)
