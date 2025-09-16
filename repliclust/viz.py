"""
Provides the built-in visualization features of `repliclust`.

**Functions**:
    :py:func:`plot`
        Plot a dataset.
"""

import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


def plot(X, y=None, dimensionality_reduction="tsne", dim_red_params={}, **plot_params):
    """
    Plot high-dimensional data with dimensionality reduction and clustering labels.

    This function creates a 2D scatter plot of the input data `X`. If `X` has more than two
    features, dimensionality reduction is performed using either t-SNE or UMAP before plotting.
    Optionally, data points can be colored according to cluster labels provided in `y`.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        The input data to plot.

    y : array-like of shape (n_samples,), optional
        Cluster labels or target values used to color the data points. If `None`, all points
        are plotted with the same color.

    dimensionality_reduction : {'tsne', 'umap'}, default='tsne'
        The method used for dimensionality reduction when `X` has more than two features.
        Choices are:

        - 'tsne' : Use t-distributed Stochastic Neighbor Embedding.
        - 'umap' : Use Uniform Manifold Approximation and Projection.

    dim_red_params : dict, default={}
        Additional keyword arguments to pass to the dimensionality reduction algorithm.

    **plot_params
        Additional keyword arguments passed to `matplotlib.pyplot.scatter`.

    Raises
    ------
    ValueError
        If `X` has fewer than two features.

    ValueError
        If `dimensionality_reduction` is not one of 'tsne' or 'umap'.

    See Also
    --------
    matplotlib.pyplot.scatter : Create a scatter plot.
    sklearn.manifold.TSNE : t-distributed Stochastic Neighbor Embedding.
    umap.UMAP : Uniform Manifold Approximation and Projection.

    Examples
    --------
    Plot data with t-SNE dimensionality reduction:

    >>> plot(X, y, dimensionality_reduction='tsne')

    Plot data with UMAP dimensionality reduction and custom parameters:

    >>> dim_red_params = {'n_neighbors': 15, 'min_dist': 0.1}
    >>> plot(X, y, dimensionality_reduction='umap', dim_red_params=dim_red_params)

    Plot 2D data without dimensionality reduction:

    >>> X_2d = np.random.rand(100, 2)
    >>> plot(X_2d, y)
    """
    plt.figure()

    if X.shape[1] == 2:
        T = X

    elif X.shape[1] > 2:
        if dimensionality_reduction=="tsne":
            tsne_model = TSNE(n_components=2, perplexity=30, **dim_red_params)
            T = tsne_model.fit_transform(X)
        elif dimensionality_reduction=="umap":
            try:
                import umap
            except Exception as e:
                raise ImportError(
                    "UMAP plotting requires the 'umap-learn' package and compatible dependencies."
                ) from e
            umap_model = umap.UMAP(n_neighbors=30, n_components=2, **dim_red_params)
            T = umap_model.fit_transform(X)
        else:
            raise ValueError(
                "dimensionality_reduction should be one of 'tsne' or 'umap'" 
                + f" (found '{dimensionality_reduction}')"
            )

    elif X.shape[1] < 2:
        raise ValueError(f"dimensionality must be >=2 (found '{X.shape[1]}')")

    plt.scatter(T[:,0], T[:,1], c=y, **plot_params)
    
    if X.shape[1] == 2:
        plt.xlabel("X1")
        plt.ylabel("X2")
    elif dimensionality_reduction=="tsne":
        plt.xlabel("TSNE1")
        plt.ylabel("TSNE2")
    elif dimensionality_reduction=="umap":
        plt.xlabel("UMAP1")
        plt.ylabel("UMAP2")
    
    plt.show()