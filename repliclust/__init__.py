"""
Synthetic Data Generator for Cluster Analysis
=============================================

repliclust is a Python module allowing reproducible synthetic data
generation for cluster analysis. 

The module is based on data blueprints, which allow constrained sampling
of synthetic data sets with similar geometries. For example, the user
may wish to sample many different data sets that each contain 22 
slightly oblong clusters that are barely touching.

See http://github.com/mzelling/repliclust for the project webpage.

"""

import numpy as np

from repliclust import config
from repliclust.base import set_seed, SUPPORTED_DISTRIBUTIONS

config.init_rng()

# indicates public API, specifies which submodules to import when
# running "from repliclust import *"
__all__ = [
    'base',
    'overlap'
    'maxmin',
    'distributions',
]