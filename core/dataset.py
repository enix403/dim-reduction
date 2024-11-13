import torch as tr
from matplotlib import pyplot as plt
from sklearn.datasets import make_blobs

def _make_clustered_dataset(n, dim, num_classes):
    X, labels = make_blobs(
        n_samples=n,
        n_features=dim,
        centers=num_classes,
        random_state=1601
    )

    X = tr.tensor(X, dtype=tr.float32, requires_grad=False)
    labels = tr.tensor(labels, dtype=tr.long, requires_grad=False)

    return X, labels

dim_high = 50 # H
num_samples = 100 # N
num_classes = 3 # C

X, labels = _make_clustered_dataset(num_samples, dim_high, num_classes)

def plot_reduced(points):
    n_dim = points.shape[1]

    if n_dim != 2 and n_dim != 3:
        raise Exception("Data must be 2d or 3d")

    comps = tr.unbind(points, dim=1)

    ax = plt.subplot(projection="3d" if n_dim == 3 else "rectilinear")
    ax.scatter(*comps, c=labels, cmap="viridis")
    plt.show()