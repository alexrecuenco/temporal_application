from kmeans import KMeans


from random import random

N_CLUSTER = 10
N_POINTS = 100
N_DIMENS = 2


def rand_point(n_dimens):
    return tuple(random() for _ in range(n_dimens))


def main():
    # Actually testing if the implementation works, would require to test agains ta current model...
    # Here we are just asserting that it doesn't do anything crazy, and that it terminates
    points = list(rand_point(N_DIMENS) for _ in range(100))
    kmeans = KMeans(N_CLUSTER).fit(points)

    predictions = kmeans.predict(points)

    assert all(isinstance(pred, int) and 0 <= pred < N_CLUSTER for pred in predictions)
    print("Looks, ok? ...")


if __name__ == "__main__":
    main()
