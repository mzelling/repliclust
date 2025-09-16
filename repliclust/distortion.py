"""
Provides functions for nonlinearly distorting datasets, so that clusters
become non-convex and take on more irregular shapes beyond ellipsoids.

**Functions**:
    :py:func:`distort`
        Distort a dataset.
    :py:func:`project_to_sphere`
        Apply stereographic projection to make the data directional.
"""

import numpy as np
try:
    import torch  # optional
    from torch import nn
except Exception:  # ImportError when torch missing
    torch = None  # type: ignore
    class nn:  # type: ignore
        Module = object
from scipy.stats import ortho_group


def construct_near_ortho_matrix(hidden_dim, scaling_factor=0.1):
    """
    Construct a near-orthogonal matrix.

    Generates a random near-orthogonal matrix of size `hidden_dim` x `hidden_dim`
    by perturbing an orthogonal matrix.

    Parameters
    ----------
    hidden_dim : int
        The dimension of the square matrix to generate.

    scaling_factor : float, optional
        Standard deviation of the normal distribution used to generate logarithms
        of scaling factors. Defaults to 0.1.

    Returns
    -------
    torch.Tensor
        A `hidden_dim` x `hidden_dim` near-orthogonal matrix of type `torch.float32`.

    Notes
    -----
    The generated matrix is obtained by scaling the eigenvalues of a random orthogonal
    matrix, ensuring the determinant remains ±1.

    """
    axes = ortho_group.rvs(hidden_dim)
    logs = np.random.normal(loc=0, scale=scaling_factor, size=hidden_dim)
    scalings = np.exp(logs - np.mean(logs))
    sign = np.random.choice(a=[-1,1], p=[0.5,0.5])
    assert np.allclose(np.prod(scalings), 1.0)
    return sign * torch.tensor(np.transpose(axes) @ np.diag(scalings) @ axes, dtype=torch.float32)


class NeuralNetwork(nn.Module):
    """
    Random neural network for data distortion.

    This neural network applies a series of linear and non-linear transformations to input data,
    intended to distort datasets and make clusters take on irregular shapes.

    Parameters
    ----------
    hidden_dim : int, optional
        The dimensionality of the hidden layers. Defaults to 64.

    dim : int, optional
        The input and output dimensionality of the data. Defaults to 2.

    n_layers : int, optional
        The number of hidden layers in the network. Defaults to 50.

    """
    def __init__(self, hidden_dim=64, dim=2, n_layers=50):
        if torch is None:
            raise ImportError(
                "repliclust 'distort' requires the 'distort' extra:\n"
                "  pip install repliclust[distort]"
            )
        super().__init__()

        embedding = nn.Linear(dim, hidden_dim)
        projection = nn.Linear(hidden_dim, dim)
        # Uncomment line below if you want to tie the projection and embeddings weights:
        projection.weight = nn.Parameter(embedding.weight.T)

        middle_layers = []
        for i in range(n_layers):
            layer = nn.Linear(hidden_dim, hidden_dim)
            middle_layers.append(layer)
            middle_layers.append(nn.LayerNorm(hidden_dim))
            middle_layers.append(nn.Tanh())

        self.middle_stack = nn.Sequential(
            embedding,
            *(middle_layers),
            projection
        )

    def forward(self, x):
        """
        Forward pass of the neural network.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (n_samples, dim).

        Returns
        -------
        torch.Tensor
            Transformed tensor of shape (n_samples, dim).

        """
        x_transformed = self.middle_stack(x)
        return x_transformed
    

def distort(X, hidden_dim=128, n_layers=16, device="cuda", set_seed=None):
    """
    Distort a dataset using a random neural network.

    Transforms the input dataset `X` by passing it through a randomly initialized neural network,
    causing clusters to take on irregular, non-convex shapes.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        The input data to be distorted.

    hidden_dim : int, optional
        The dimensionality of the hidden layers in the neural network. Defaults to 128.

    n_layers : int, optional
        The number of hidden layers in the neural network. Defaults to 16.

    device : {'cuda', 'cpu'}, optional
        The device on which to perform computations. Defaults to 'cuda'.

    set_seed : int or None, optional
        Random seed for reproducibility. If `None`, the random seed is not set.

    Returns
    -------
    torch.Tensor
        The distorted data as a tensor of shape (n_samples, n_features).

    Notes
    -----
    If CUDA is not available, the device will be automatically switched to 'cpu'.

    Examples
    --------
    Distort a dataset and convert the result to a NumPy array:

    >>> X_distorted = distort(X).numpy()

    """
    if torch is None:
        raise ImportError(
            "repliclust 'distort' requires the 'distort' extra:\n"
            "  pip install repliclust[distort]"
        )
    if not torch.cuda.is_available():
        device = "cpu"
        print("Switched to CPU because CUDA is not available.")

    if set_seed is not None:
        torch.manual_seed(set_seed)

    dim = X.shape[1]
    random_nn = NeuralNetwork(hidden_dim=hidden_dim, dim=dim, n_layers=n_layers).to(device)

    max_length = np.sqrt(np.max(np.sum(X**2,axis=1)))
    X_norm = X/max_length

    with torch.no_grad():
        X_tensor = torch.tensor(X_norm.astype('float32')).to(device)
        X_tf = random_nn(X_tensor).cpu()
    
    return X_tf.numpy()


def wrap_around_sphere(X):
    """
    Apply inverse stereographic projection to data.

    Projects the input data `X` onto the unit sphere using inverse stereographic
    projection, making the data directional.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        The input data to be projected.

    Returns
    -------
    ndarray of shape (n_samples, n_features + 1)
        The projected data lying on the unit sphere.

    Notes
    -----
    The inverse stereographic projection maps points from Euclidean space onto 
    the sphere. The output data will have one additional dimension compared to 
    the input.

    Examples
    --------
    Project a 2D dataset onto the sphere:

    >>> X_spherical = wrap_around_sphere(X)

    Verify that the projected points lie on the unit sphere:

    >>> np.allclose(np.linalg.norm(X_spherical, axis=1), 1.0)
    True

    """
    lengths = np.sqrt(np.sum(X**2,axis=1))
    l_max = np.max(lengths)

    # normalize data so that the maximum length is 1
    X_tf = X/l_max

    # carry out the inverse stereographic projection to yield x_full on the sphere
    partial_squared_norms = np.sum(X_tf**2, axis=1)
    x_p = (1 - partial_squared_norms) / (1 + partial_squared_norms)
    x_rest = X_tf * (1 + x_p[:, np.newaxis])
    x_full = np.concatenate([x_rest, x_p[:, np.newaxis]], axis=1)
    assert np.allclose(np.sum(x_full**2, axis=1), 1.0)

    return x_full