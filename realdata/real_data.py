"""
Real-dataset federated partitioner. Produces clients in the exact
(X, y) list format make_federated_data() returns, so it is a drop-in
replacement for the FedProx solver already validated in simulation.py.

Two real datasets:
  - 'digits': sklearn's bundled load_digits (1,797 samples, 8x8=64 dims,
    10 classes). Zero network dependency.
  - 'mnist': subsampled real MNIST via a pre-downloaded local cache
    (mnist_full.npz), 28x28=784 dims, 10 classes.

Partitioning: Dirichlet class-proportion skew, identical in spirit to
make_federated_data()'s synthetic partitioner (same alpha parameter,
same meaning: lower alpha = more heterogeneous). Sampling is WITH
replacement from each class's real pool, since under skewed alpha some
clients draw disproportionately from rare classes and without-replacement
sampling can exhaust a small pool; this is standard practice in the
federated-partition literature (e.g. LEAF-style harnesses).
"""
import numpy as np


def _load_digits():
    from sklearn.datasets import load_digits
    d = load_digits()
    X = d.data.astype(np.float64)   # (1797, 64), values 0-16
    y = d.target.astype(int)
    X = X / X.max()                  # normalise to [0, 1]
    return X, y, 10, 64


def _load_mnist(pool_size=10000, seed=0):
    d = np.load("mnist_full.npz")
    X_full, y_full = d["X"], d["y"]
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X_full), size=pool_size, replace=False)
    X = X_full[idx].astype(np.float64) / 255.0   # normalise to [0, 1]
    y = y_full[idx].astype(int)
    return X, y, 10, 784


def make_real_federated_data(dataset, N=100, n_per=50, alpha=0.3, seed=0):
    """
    dataset : 'digits' or 'mnist'
    Returns (clients, K, dim) exactly like make_federated_data().
    """
    rng = np.random.default_rng(seed)

    if dataset == "digits":
        X, y, K, dim = _load_digits()
    elif dataset == "mnist":
        X, y, K, dim = _load_mnist(seed=seed)
    else:
        raise ValueError(dataset)

    by_class = [np.where(y == k)[0] for k in range(K)]
    min_pool = min(len(c) for c in by_class)
    if min_pool == 0:
        raise RuntimeError("a class has zero real examples in the pool")

    props = rng.dirichlet([alpha] * K, size=N)

    clients = []
    for i in range(N):
        counts = rng.multinomial(n_per, props[i])
        X_parts, y_parts = [], []
        for k in range(K):
            if counts[k] > 0:
                chosen = rng.choice(by_class[k], size=counts[k], replace=True)
                X_parts.append(X[chosen])
                y_parts.extend([k] * counts[k])
        Xc = np.vstack(X_parts) if X_parts else np.zeros((0, dim))
        yc = np.array(y_parts, dtype=int)
        clients.append((Xc, yc))

    return clients, K, dim
