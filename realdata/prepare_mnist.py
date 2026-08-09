"""
One-time conversion of the standard MNIST IDX training files into the
mnist_full.npz format real_data.py expects.

Run once, from within realdata/:

    python prepare_mnist.py

The two .gz inputs are the unmodified training-set files from the
original MNIST distribution (Yann LeCun / Corinna Cortes / Christopher
Burges), 60,000 28x28 greyscale digit images with labels. Only the
training split is used; Section 8.9 draws a 10,000-image subsample from
it (see real_data.py, make_real_federated_data).
"""
import gzip
import struct
import numpy as np


def _read_idx_images(path):
    with gzip.open(path, "rb") as f:
        magic, n, rows, cols = struct.unpack(">IIII", f.read(16))
        assert magic == 2051, f"unexpected magic number {magic} in {path}"
        buf = f.read(n * rows * cols)
        return np.frombuffer(buf, dtype=np.uint8).reshape(n, rows * cols)


def _read_idx_labels(path):
    with gzip.open(path, "rb") as f:
        magic, n = struct.unpack(">II", f.read(8))
        assert magic == 2049, f"unexpected magic number {magic} in {path}"
        buf = f.read(n)
        return np.frombuffer(buf, dtype=np.uint8)


if __name__ == "__main__":
    X = _read_idx_images("train-images-idx3-ubyte.gz")
    y = _read_idx_labels("train-labels-idx1-ubyte.gz")
    assert X.shape == (60000, 784) and y.shape == (60000,)
    np.savez_compressed("mnist_full.npz", X=X, y=y)
    print(f"Wrote mnist_full.npz: X {X.shape} {X.dtype}, y {y.shape} {y.dtype}")
