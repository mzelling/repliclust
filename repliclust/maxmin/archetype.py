"""
This module implements a archetype for mixture models. The user chooses
the desired geometry by setting the ratios between largest and smallest
values of various geometric parameters.
"""

import numpy as np
import copy
import json

from repliclust import config as CONFIG
from repliclust.base import Archetype
from repliclust.overlap.centers import ConstrainedOverlapCenters

from repliclust.maxmin.covariance import MaxMinCovarianceSampler
from repliclust.maxmin.groupsizes import MaxMinGroupSizeSampler
from repliclust.distributions import FixedProportionMix


def validate_overlaps(max_overlap=0.05,min_overlap=1e-3):
    """ 
    Note that we allow max_overlap=1 and min_overlap=0, which
    should have the effect of removing one or both overlap constraints.
    """
    if (max_overlap <= 0):
        raise ValueError('max_overlap must be greater than zero because'
            + ' clusters are based on probability distributions that '
            + ' always overlap a small -- potentially tiny -- amount.'
            + ' Try setting max_overlap close to zero to achieve large'
            + ' separation between clusters.')
    elif (min_overlap >= 1):
        raise ValueError('min_overlap must be smaller than 1 because we'
            + ' cannot force all clusters to overlap COMPLETELY. Try'
            + ' setting min_overlap to a value close to 1 instead.')
    elif (max_overlap > 1) or (min_overlap < 0):
        raise ValueError("max_overlap and min_overlap should be" +
                         " between 0 and 1.")
    elif max_overlap <= min_overlap:
        raise ValueError("max_overlap must exceed min_overlap.")

def validate_maxmin_ratios(maxmin_ratio=2, arg_name="aspect_maxmin",
                            underlying_param="aspect ratio"):
    """ Check that a max-min ratio is >= 1. """
    if (maxmin_ratio < 1):
        raise ValueError("the parameter " + arg_name 
                + " must be >= 1, as it is the ratio"
                + " of the maximum " + underlying_param
                + " to the minimum " + underlying_param)

def validate_reference_quantity(ref_qty=1.5, min_allowed_value=1, 
                                name="aspect_ref"):
    """ 
    Check that a reference value exceeds its minimum allowed value. 
    """
    if (ref_qty < min_allowed_value):
        raise ValueError("the parameter " + name
                            + " should be greater than "
                            + str(min_allowed_value))
    

def validate_archetype_args(**args):
    """
    Validate all provided arguments for a MaxMinArchetype.

    This function checks the validity of overlap parameters, max-min ratios, and reference quantities.

    Parameters
    ----------
    **args
        Arbitrary keyword arguments containing parameters to validate.

    Raises
    ------
    ValueError
        If any of the parameters fail their respective validations.
    """
    validate_overlaps(args['max_overlap'], args['min_overlap'])

    maxmin_args = [
        ('aspect_maxmin', 
            args['aspect_maxmin'], 'cluster aspect ratio'),
        ('radius_maxmin', 
            args['radius_maxmin'], 'cluster radius'),
        ('imbalance_ratio', 
            args['imbalance_ratio'], 'cluster group size'),
        ]
    for arg_name, arg_val, underlying_param in maxmin_args:
        validate_maxmin_ratios(maxmin_ratio=arg_val, arg_name=arg_name,
                               underlying_param=underlying_param)

    ref_args = [
        ("aspect_ref", args['aspect_ref'], 1),
        # ("scale", args['scale'], 0),
    ]
    for arg_name, arg_val, min_allowed_val in ref_args:
        validate_reference_quantity(ref_qty=arg_val, 
                                    min_allowed_value=min_allowed_val,
                                    name=arg_name)


def parse_distribution_selection(distributions: list, proportions=None):
    """
    Parse user selection of probability distributions.

    Reformats the user-provided list of distributions and proportions into a format suitable for constructing a `FixedProportionMix` object.

    Parameters
    ----------
    distributions : list of str or tuple
        Selection of probability distributions to include in each mixture model.
        Each element is either:
        - A string representing the name of the distribution.
        - A tuple `(name, params_dict)`, where `name` is the distribution name and `params_dict` is a dictionary of distribution parameters.

    proportions : list of float, optional
        Proportions of clusters that should have each distribution. If `None`, distributions are equally weighted.

    Returns
    -------
    list of tuple
        A list suitable for constructing a `FixedProportionMix` object, where each element is a tuple `(name, proportion, params_dict)`.

    Raises
    ------
    ValueError
        If `distributions` is not in the expected format.

    Notes
    -----
    To print all valid distribution names, call `repliclust.print_supported_distributions()`.
    """
    if not proportions:
        proportions = np.full(len(distributions), 
                                fill_value=1/len(distributions))

    distr_parsed = []
    for i, distr in enumerate(distributions):
        distr_prop = proportions[i]
        if isinstance(distr, tuple) and isinstance(distr[1], dict):
            distr_name = distr[0]
            distr_params = distr[1]
            distr_parsed.append((distr_name, distr_prop, distr_params))
        elif isinstance(distr, str):
            distr_parsed.append((distr, distr_prop, {}))
        else:
            raise ValueError("distributions should be provided as a"
                    + " list of which each element is either a string"
                    + " or a two-element tuple whose first element is"
                    + " a string and whose second element is a"
                    + " dictionary")
    return distr_parsed




class MaxMinArchetype(Archetype):
    """
    A dataset archetype that defines the overall geometry using max-min ratios.

    The user sets the ratios between largest and smallest values of various geometric parameters.

    Parameters
    ----------
    n_clusters : int
        The desired number of clusters.

    dim : int
        The desired number of dimensions.

    n_samples : int
        Total number of samples in the dataset.

    max_overlap : float, optional
        Maximum allowed overlap between any two clusters, as a fraction between 0 and 1. Default is 0.05.

    min_overlap : float, optional
        Minimum required overlap between a cluster and some other cluster, as a fraction between 0 and 1. Default is 1e-3.

    imbalance_ratio : float, optional
        Ratio between the largest and smallest group sizes among clusters. Must be >= 1. Default is 2.

    aspect_maxmin : float, optional
        Ratio between the largest and smallest aspect ratios among clusters. Must be >= 1. Default is 2.

    radius_maxmin : float, optional
        Ratio between the largest and smallest radii among clusters. Must be >= 1. Default is 3.

    aspect_ref : float, optional
        Reference aspect ratio for clusters. Must be >= 1. Default is 1.5.

    name : str, optional
        Name of the archetype. If `None`, a default name is assigned.

    scale : float, optional
        Reference length scale for generated data. Default is 1.0.

    packing : float, optional
        Packing density parameter affecting cluster placement. Default is 0.1.

    distributions : list of str or tuple, optional
        Selection of probability distributions for the clusters. Default is `['normal', 'exponential']`.

    distribution_proportions : list of float, optional
        Proportions of clusters that should have each distribution.

    overlap_mode : {'auto', 'lda', 'c2c'}, optional
        Degree of precision when computing cluster overlaps: 'lda' is more exact
         than 'c2c' but more computationally expensive. Default is 'auto', which
         switches automatically switches from 'lda' to 'c2c'.

    linear_penalty_weight : float, optional
        Weight of the linear penalty in the overlap optimization. Default is 0.01.

    learning_rate : float or 'auto', optional
        Learning rate for overlap optimization. If 'auto', it is set based on the dimensionality. Default is 'auto'.

    Notes
    -----
    **Glossary of geometric terms:**

    - **Group size** : Number of data points in a cluster.
    - **Cluster radius** : Geometric mean of the standard deviations along a cluster's principal axes.
    - **Cluster aspect ratio** : Ratio between the lengths of the longest and shortest principal axes of a cluster.

    Examples
    --------
    Create an archetype with default parameters:

    >>> archetype = MaxMinArchetype()

    Create an archetype with specific parameters:

    >>> archetype = MaxMinArchetype(n_clusters=10, dim=5, aspect_ref=2.0)
    """

    def guess_learning_rate(self, dim):
        """
        Estimate an appropriate learning rate based on the dimensionality.

        Parameters
        ----------
        dim : int
            The dimensionality of the data.

        Returns
        -------
        float
            Suggested learning rate.
        """
        return 0.5 #0.5*(1/np.log10(10+dim))

    def edit_params(self, suffix=None, **params):
        """
        Create a modified copy of this archetype with updated parameters.

        Parameters
        ----------
        suffix : str, optional
            Suffix to append to the archetype's name. If `None`, the suffix 
            'edited' is used.

        **params
            Arbitrary keyword arguments representing parameters to update.

        Returns
        -------
        MaxMinArchetype
            A new `MaxMinArchetype` instance with updated parameters.
        """
        new_params = copy.deepcopy(self.params)
        for k,v in params.items():
            new_params[k] = v
        # Until implement a natural language based renaming function,
        # simply note that the archetype has been edited
        if suffix is None:
            suffix = "edited"
        new_params["name"] = self.params["name"] + "_" + suffix
        return MaxMinArchetype(**new_params)

    def describe(self, exclude_internal_params=True):
        """
        Get a dictionary describing the archetype's parameters.

        Returns
        -------
        dict
            A dictionary containing the archetype's parameters.
        """
        if exclude_internal_params:
            return { 
                k: v for k, v in self.params.items() if k not in 
                    { 'packing', 'learning_rate', 'linear_penalty_weight' } 
            }
        else:
            return self.params
    
    def create_another_like_this(self, suffix=None, **new_params):
        """
        Create a new archetype similar to this one with updated parameters.

        Parameters
        ----------
        suffix : str, optional
            Suffix to append to the new archetype's name.

        **new_params
            Arbitrary keyword arguments representing parameters to update.

        Returns
        -------
        MaxMinArchetype
            A new `MaxMinArchetype` instance with updated parameters.
        """
        new_arch = self.edit_params(**new_params, suffix=suffix)
        return new_arch
    
    def sample_hyperparams(
            self, n=10, 
            min_n_clusters=1, max_n_clusters=30,
            min_samples_per_cluster=10, max_samples_per_cluster=1000,
            min_dim=2, max_dim=50
        ):
        """
        Generate multiple copies of this archetype by sampling hyperparameters.

        Parameters
        ----------
        n : int, optional
            Number of archetype copies to generate. Default is 10.

        min_n_clusters : int, optional
            Minimum number of clusters. Default is 1.

        max_n_clusters : int, optional
            Maximum number of clusters. Default is 30.

        min_samples_per_cluster : int, optional
            Minimum number of samples per cluster. Default is 10.

        max_samples_per_cluster : int, optional
            Maximum number of samples per cluster. Default is 1000.

        min_dim : int, optional
            Minimum dimensionality. Default is 2.

        max_dim : int, optional
            Maximum dimensionality. Default is 50.

        Returns
        -------
        list of MaxMinArchetype
            A list of new `MaxMinArchetype` instances with sampled hyperparameters.

        Notes
        -----
        The hyperparameters `n_clusters`, `n_samples`, and `dim` are sampled from Poisson distributions centered around the current archetype's parameters.
        """
        def sample_until(f, xmin=None, xmax=None, **params):
            y = 0
            while ((y < xmin) or (y > xmax)):
                y = f(**params)
            return y
        return [ 
            self.create_another_like_this(
                n_clusters=sample_until(np.random.poisson, min_n_clusters, max_n_clusters, lam=self.n_clusters),
                n_samples=sample_until(np.random.poisson, self.n_clusters*min_samples_per_cluster, self.n_clusters*max_samples_per_cluster, lam=self.n_samples),
                dim=sample_until(np.random.poisson, min_dim, max_dim, lam=self.dim),
                suffix=f"sampled_{i}"
            ) for i in range(n) 
        ]

    def __init__(
            self, 
            n_clusters=6, dim=2, n_samples=500,
            max_overlap=0.05, min_overlap=1e-3, 
            imbalance_ratio=2, aspect_maxmin=2, radius_maxmin=3,
            aspect_ref=1.5, name=None, scale=1.0, packing=0.1,
            distributions=['normal', 'exponential'],
            distribution_proportions=None,
            overlap_mode='auto', linear_penalty_weight=0.01, 
            learning_rate='auto',
            ):
        """ Instantiate a MaxMinArchetype object. """
        covariance_args = {'aspect_ref': aspect_ref,
                           'aspect_maxmin': aspect_maxmin,
                           'radius_maxmin': radius_maxmin}
        groupsize_args = {'imbalance_ratio': imbalance_ratio}

        if learning_rate=='auto':
            learning_rate = self.guess_learning_rate(dim)
        elif not isinstance(learning_rate, float):
            raise ValueError("learning_rate should be 'auto' or a "
                                + "float between 0 and 1")
        
        center_args = {'max_overlap': max_overlap, 
                       'min_overlap': min_overlap,
                       'packing': packing,
                       'learning_rate': learning_rate,
                       'linear_penalty_weight': linear_penalty_weight}
        distributions_parsed = parse_distribution_selection(
                                    distributions, 
                                    distribution_proportions)

        validate_archetype_args(**(covariance_args | groupsize_args 
                                                | center_args))
        self.params = (
            { 
                "n_clusters": n_clusters, 
                "dim": dim, 
                "n_samples": n_samples, 
                "scale": scale,
                "name": name,
                "distributions": distributions,
                "distribution_proportions": distribution_proportions
            } 
            | covariance_args 
            | groupsize_args 
            | center_args
        )

        # choose cluster center sampler
        if overlap_mode=='auto':
            overlap_mode = 'lda' if n_clusters*dim <= 10000 else 'c2c'
        
        center_sampler = ConstrainedOverlapCenters(
                            overlap_mode=overlap_mode, **center_args
                            )

        distribution_sampler = FixedProportionMix(distributions_parsed)

        Archetype.__init__(
            self, n_clusters, dim, n_samples, name, scale,
            MaxMinCovarianceSampler(**covariance_args),
            center_sampler,
            MaxMinGroupSizeSampler(**groupsize_args),
            distribution_sampler,
            **covariance_args, **groupsize_args, **center_args,
            )
    
    @staticmethod
    def from_verbal_description(description: str, name=None, openai_api_key=None):
        """
        Instantiate a `MaxMinArchetype` from a verbal description.

        Parameters
        ----------
        description : str
            Verbal description of the desired dataset archetype.

        name : str, optional
            Name to assign to the new archetype. If `None`, a name is generated.

        openai_api_key : str, optional
            OpenAI API key for accessing the language model. If `None`, the key is read from the configuration.

        Returns
        -------
        MaxMinArchetype
            A new `MaxMinArchetype` instance based on the verbal description.

        Raises
        ------
        Exception
            If the OpenAI client cannot be initialized or the archetype cannot be created.

        Notes
        -----
        This method uses a language model to parse the verbal description and generate the archetype parameters.
        """
        try:
            import repliclust.natural_language as nl  # type: ignore
        except Exception:
            raise ImportError(
                "Natural-language archetype creation requires the 'nlp' extra:\n"
                "  pip install repliclust[nlp]"
            )
        if (nl.OPENAI_CLIENT is None) and (openai_api_key is not None):
            nl.load_openai_client(api_key=openai_api_key)
        if nl.OPENAI_CLIENT is None:
            raise Exception(
                "Failed to initialize OpenAI client." +
                " Either put OPENAI_API_KEY=<...> into the .env file and reload the module" +
                " or pass openai_api_key=<...> as an argument in the function call."
            )
        
        archetype_query = nl.OPENAI_CLIENT.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": nl.MAKE_ARCHETYPE_SYSTEM_PROMPT,
                },
                {
                    "role": "user", 
                    "content": nl.MAKE_ARCHETYPE_PROMPT_TEMPLATE.format(description=description)
                }
            ],
            temperature=0.0,
            response_format= { "type": "json_object" },
            seed=CONFIG._seed,
        )
        arch_json = archetype_query.choices[0].message.content

        if name is None:
            name_query = nl.OPENAI_CLIENT.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": nl.MAKE_ARCHETYPE_NAME_SYSTEM_PROMPT,
                    },
                    {
                        "role": "user", 
                        "content": nl.MAKE_ARCHETYPE_NAME_PROMPT_TEMPLATE.format(description=description)
                    }
                ],
                temperature=0.0,
                seed=CONFIG._seed,
            )
            name_str = name_query.choices[0].message.content

        try:
            arch_json = json.loads(arch_json)
            arch_json["name"] = name_str if name is None else name
            return MaxMinArchetype(**arch_json)
        except Exception:
            print(arch_json)
            raise Exception("Failed to process this data set archetype. Please rephrase and try again.")
        
        