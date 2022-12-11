import pytest
import numpy as np

from repliclust.maxmin.blueprint import MaxMinBlueprint
from repliclust.base import MixtureModel

class TestMaxMinBlueprint():
    def test_init(self):
        # test that the object has all required attributes
        bp_with_default_args = MaxMinBlueprint()

        # test that we properly validate the parameters
        with pytest.raises(ValueError):
            # wanting zero max overlap
            MaxMinBlueprint(max_overlap=0)
        with pytest.raises(ValueError):
            # wanting negative max overlap
            MaxMinBlueprint(max_overlap=-0.1)
        with pytest.raises(ValueError):
            # wanting negative min overlap
            MaxMinBlueprint(min_overlap=-0.1)
        with pytest.raises(ValueError):
            # wanting imbalance ratio less than 1
            MaxMinBlueprint(imbalance_ratio=0.6)
        with pytest.raises(ValueError):
            # wanting imbalance ratio of 0
            MaxMinBlueprint(imbalance_ratio=0)
        with pytest.raises(ValueError):
            # wanting negative imbalance ratio
            MaxMinBlueprint(imbalance_ratio=-1)

        # creating a blueprint with zero minimum overlap should work
        zero_min_overlap = MaxMinBlueprint(min_overlap=0)

        # imbalance ratio of 1 should work
        imbal_ratio_unity = MaxMinBlueprint(imbalance_ratio=1)

    def test_sample_mixture_model(self):
        bp = MaxMinBlueprint(n_clusters=30, dim=2,
                             max_overlap=0.05, min_overlap=0.04)
        mixture_model = bp.sample_mixture_model()

        # make sure we get a mixture model
        assert isinstance(mixture_model, MixtureModel)