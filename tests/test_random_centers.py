import pytest
import numpy as np

import repliclust
from repliclust.base import Archetype
from repliclust.random_centers import adjusted_log_packing
from repliclust.random_centers import RandomCenters

def test_adjusted_log_packing():
    with pytest.raises(ValueError):
        # provide invalid packing
        adjusted_log_packing(packing=0,dim=1)
    with pytest.raises(ValueError):
        # provide dimension less than 2
        adjusted_log_packing(packing=0.5,dim=1)
    
    assert np.allclose(adjusted_log_packing(packing=0.1, dim=2),
                        np.log(0.1))
    assert np.allclose(adjusted_log_packing(packing=1, dim=10),
                        0 + np.log(9) - 8*np.log(2))
    assert np.allclose(adjusted_log_packing(packing=5, dim=100),
                        np.log(5) + np.log(99) - 98*np.log(2))

def test_RandomCenters():
    with pytest.raises(ValueError):
        # attempt to create object with packing=0
        rnd_centers = RandomCenters(packing=0)

    rnd_centers = RandomCenters(packing=2)
    # Try scale = 0 -> clusters have zero volume, and
    # sampling box will also have zero volume.
    arch = Archetype(n_clusters=100, dim=3, scale=0)
    centers = rnd_centers.sample_cluster_centers(arch)
    assert np.allclose(centers, 0)
    assert centers.shape == (100, 3)

    # Test with packing=1 in 100 dim
    rnd_centers = RandomCenters(packing=1)
    arch = Archetype(n_clusters=1, dim=100, scale=0.1)
    centers = rnd_centers.sample_cluster_centers(arch)
    assert centers.shape == (1, 100)

    # Test with packing=0.5 in 2D
    rnd_centers = RandomCenters(packing=0.5)
    arch = Archetype(n_clusters=7, dim=2, scale=1)
    centers = rnd_centers.sample_cluster_centers(arch)
    print(centers)
    assert centers.shape == (7, 2)

    # Test random seed
    repliclust.set_seed(11)
    centers1 = rnd_centers.sample_cluster_centers(arch)
    centers2 = rnd_centers.sample_cluster_centers(arch)
    repliclust.set_seed(11)
    centers3 = rnd_centers.sample_cluster_centers(arch)
    assert (not np.allclose(centers1,centers2))
    assert np.allclose(centers1, centers3)
