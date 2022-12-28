"""
repliclust
==========

`repliclust` is a Python package for generating synthetic data sets 
with clusters. 

The package is based on data set archetypes, high-level geometric 
blueprints that allow you to sample many data sets with the same overall
geometric structure. The following modules and subpackages are available.

**Modules**:
    `repliclust.base`
        Provides the core framework of `repliclust`.
    `repliclust.distributions`
        Implements probability distributions and related functionality.

**Subpackages**:
    `repliclust.maxmin`
        Implements a data set archetype based on max-min ratios.
    `repliclust.overlap`
        Helps locate cluster centers with the desired overlap.
"""

import numpy as np

from repliclust import config
from repliclust.base import set_seed, SUPPORTED_DISTRIBUTIONS
from repliclust import base, overlap, maxmin, distributions
from repliclust.base import DataGenerator, get_supported_distributions
from repliclust.maxmin import MaxMinArchetype as Archetype

config.init_rng()

# Indicate which components to import with "from repliclust import *"
__all__ = [
    'base',
    'overlap'
    'maxmin',
    'distributions',
]