# Question

Given a set of two dimensional points `P (e.g. [(1.1, 2.5), (3.4,1.9)...]`; the size of set can be 100s), write a function that calculates simple `K`-means. The expected returned value from the function is 

1. a set of cluster id that each point belongs to, and 
2. coordinates of centroids at the end of iteration.

Although you can write this in any language, we would recommend for you to use **python**. Please feel free to research and look up any information you need, but please note plagiarism will not be tolerated.
You may spend as much time as needed, but as a frame of reference, an hour would be the maximum time frame. If more time is required, please send over the intermediate code at the one hour mark. You may start the assignment whenever you are ready. Once you have completed this task, get back to us along with the code and how long it took you.



## Solution

1. The solution implemented works for n-dimensional clusters.
2. It implements a class Kmeans, that only has the `fit`, `fit_predict` and `predict` methods that the sklearn class considers
3. The file `test.py` tests that
4. After one hour, I had some issues with using ggplot (plotnine) to output a plot, I think it is based on column names succintness, but since the hour has passed I gave up.

## Scripts

It requires python 3.6 or above.

Both scripts don't have input arguments due to time constraints, but at the start of them they have a few variables that can be modified, `N_CLUSTER` `N_POINTS` `N_DIMENS` .



### KMeans

The main test to see "on the surface" if it seems to work.

```bash
    python test.py 
```
### Plotting

The requirements are given in the requirements.txt

```bash
    pip3 install -r requirements.txt
    python plot.py
```

This will output a png image called `plot.png`



## Thanks

Thank you for the opportunity,

Alejandro