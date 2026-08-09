"""
simulation.py
==============
Vendored, unmodified, from the companion repository for the second paper
in this series:

    "Federated Mobile Crowd Learning: Convergence, Privacy, and
     Lifecycle Carbon Guarantees"
    https://github.com/Dr-PKDP/FMCL-paper2-simulation

Included here so this repository is self-contained and does not require
cloning a second repository to reproduce Sections 7-9 of Paper 5, which
couple this FedProx engine to the thermal/charging scheduler in
converge_charging.py. See that paper for validation of this engine on its
own terms; Paper 5 uses it only as a fixed local-training subroutine.
"""

import numpy as np
from scipy.stats import norm as sp_norm


# ---------------------------------------------------------------------------
# Data generation
# ---------------------------------------------------------------------------

def make_federated_data(N=200, K=4, dim=10, n_per=50, alpha=0.3, seed=0):
    """
    Synthesise a non-IID federated dataset for N clients.

    Parameters
    ----------
    N       : number of clients
    K       : number of classes
    dim     : feature dimension
    n_per   : local samples per client
    alpha   : Dirichlet concentration (lower = more heterogeneous)
    seed    : RNG seed for reproducibility

    Returns
    -------
    clients : list of (X, y) arrays, one per client
    K       : number of classes (passed through for convenience)
    dim     : feature dimension  (passed through for convenience)
    """
    rng = np.random.default_rng(seed)

    # One random cluster centre per class
    centres = rng.standard_normal((K, dim)) * 2.0

    # Dirichlet class proportions: each client draws a mixture weight
    props = rng.dirichlet([alpha] * K, size=N)

    clients = []
    for i in range(N):
        counts = rng.multinomial(n_per, props[i])
        X_parts, y_parts = [], []
        for k in range(K):
            if counts[k] > 0:
                X_parts.append(
                    rng.standard_normal((counts[k], dim)) + centres[k]
                )
                y_parts.extend([k] * counts[k])
        X = np.vstack(X_parts) if X_parts else np.zeros((0, dim))
        y = np.array(y_parts, dtype=int)
        clients.append((X, y))

    return clients, K, dim


# ---------------------------------------------------------------------------
# Model helpers  (multinomial logistic regression)
# ---------------------------------------------------------------------------

def _softmax(Z):
    Z = Z - Z.max(axis=1, keepdims=True)
    e = np.exp(Z)
    return e / e.sum(axis=1, keepdims=True)


def _loss_and_grad(W, X, y, K):
    """Cross-entropy loss gradient.  Returns (gradient, loss)."""
    if len(X) == 0:
        return np.zeros_like(W), 0.0
    Z = X @ W
    P = _softmax(Z)
    n = len(X)
    Y = np.zeros((n, K))
    Y[np.arange(n), y] = 1.0
    grad = X.T @ (P - Y) / n
    loss = -np.log(P[np.arange(n), y] + 1e-12).mean()
    return grad, loss


def _global_grad(W, clients, K):
    """Population-weighted gradient of the global objective F(w)."""
    G = np.zeros_like(W)
    total = 0
    for X, y in clients:
        g, _ = _loss_and_grad(W, X, y, K)
        G += g * len(X)
        total += len(X)
    return G / total


def _fedprox_local(W_global, X, y, K, mu=0.1, local_steps=5, lr=0.5):
    """
    One round of FedProx local optimisation.
    Returns the update delta  W_local - W_global.
    """
    W = W_global.copy()
    for _ in range(local_steps):
        g, _ = _loss_and_grad(W, X, y, K)
        g = g + mu * (W - W_global)   # proximal regularisation
        W = W - lr * g
    return W - W_global


# ---------------------------------------------------------------------------
# Main simulation loop
# ---------------------------------------------------------------------------

def run(clients, K, dim,
        p=0.5, T=150,
        mu=0.1, clip=5.0,
        sigma=0.0, rho=0.0,
        seed=1):
    """
    Run T rounds of FedProx with stochastic (optionally correlated) dropout
    and optional Gaussian DP noise.

    Parameters
    ----------
    clients : list of (X, y) from make_federated_data()
    K, dim  : class count and feature dimension
    p       : per-round participation probability
    T       : number of training rounds
    mu      : FedProx proximal coefficient
    clip    : L2 gradient clip bound  (C in the paper)
    sigma   : DP noise multiplier  (0 = no privacy noise)
    rho     : equicorrelation coefficient for participation
              (0 = independent Bernoulli, >0 = correlated via a common factor)
    seed    : RNG seed

    Returns
    -------
    grads : array of shape (T,)  — ||nabla F(w^t)||^2 at each round
    """
    rng = np.random.default_rng(seed)
    N = len(clients)
    W = np.zeros((dim, K))
    grads = []

    for _ in range(T):

        # --- participation indicators ---
        if rho > 0:
            # equicorrelated model: A_i = 1{sqrt(rho)*Z0 + sqrt(1-rho)*eps_i > threshold}
            threshold = sp_norm.ppf(1.0 - p)
            Z0  = rng.standard_normal()
            eps = rng.standard_normal(N)
            Z   = np.sqrt(rho) * Z0 + np.sqrt(1.0 - rho) * eps
            A   = (Z > threshold).astype(float)
        else:
            A = (rng.random(N) < p).astype(float)

        # --- local updates + aggregation ---
        agg = np.zeros((dim, K))
        for i in range(N):
            if A[i] > 0:
                delta = _fedprox_local(W, clients[i][0], clients[i][1],
                                       K, mu=mu, local_steps=5, lr=0.5)
                # clip update norm to C
                nrm = np.linalg.norm(delta)
                if nrm > clip:
                    delta = delta * clip / nrm
                # add DP Gaussian noise
                if sigma > 0:
                    delta = delta + rng.standard_normal(delta.shape) * sigma * clip
                agg += A[i] * delta

        # debiased aggregation (Assumption 5 in the paper)
        W = W + agg / (N * p)

        grads.append(np.linalg.norm(_global_grad(W, clients, K)) ** 2)

    return np.array(grads)