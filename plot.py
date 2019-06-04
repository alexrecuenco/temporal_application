from plotnine import *
from kmeans import KMeans
from test import rand_point
import pandas as pd

if __name__ == "__main__":
    points = [rand_point(2) for _ in range(200)]
    kmeans = KMeans(9).fit(points)
    predictions = kmeans.predict(points)
    table = pd.DataFrame(points)
    table["tag"] = predictions
    table.rename(columns={table.columns[0]: "x", table.columns[1]: "y"}, inplace=True)
    centroids = kmeans.centroids
    plot = ggplot(aes(x="x", y="y"), table) + geom_point(aes(color="factor(tag)"))
    plot.save(filename="plot.png", height=5, width=5, units="in", dpi=300)
