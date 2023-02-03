import pytest
import numpy as np

from repliclust.maxmin.archetype import MaxMinArchetype
from repliclust.base import MixtureModel

class TestMaxMinArchetype():
    def test_init(self):
        # test that the object has all required attributes
        bp_with_default_args = MaxMinArchetype()

        # test that we properly validate the parameters
        with pytest.raises(ValueError):
            # wanting zero max overlap
            MaxMinArchetype(max_overlap=0)
        with pytest.raises(ValueError):
            # wanting negative max overlap
            MaxMinArchetype(max_overlap=-0.1)
        with pytest.raises(ValueError):
            # wanting negative min overlap
            MaxMinArchetype(min_overlap=-0.1)
        with pytest.raises(ValueError):
            # wanting imbalance ratio less than 1
            MaxMinArchetype(imbalance_ratio=0.6)
        with pytest.raises(ValueError):
            # wanting imbalance ratio of 0
            MaxMinArchetype(imbalance_ratio=0)
        with pytest.raises(ValueError):
            # wanting negative imbalance ratio
            MaxMinArchetype(imbalance_ratio=-1)

        # Should creating a archetype with zero minimum overlap work?
        # zero_min_overlap = MaxMinArchetype(min_overlap=0)

        # Imbalance ratio of 1 should work
        imbal_ratio_unity = MaxMinArchetype(imbalance_ratio=1)

    def test_sample_mixture_model(self):
        bp = MaxMinArchetype(n_clusters=30, dim=2,
                             max_overlap=0.05, min_overlap=0.04)
        mixture_model = bp.sample_mixture_model()

        # make sure we get a mixture model
        assert isinstance(mixture_model, MixtureModel)