import pytest
import numpy as np
from repliclust.overlap._gradients_marginal_lda import get_1d_idx

def test_get_1d_idx():
    # i < j
    assert np.all(get_1d_idx(0,np.array([1,2,3,4]),6) == np.array([0,1,2,3]))
    assert np.all(get_1d_idx(1,np.array([3,4,5]),6) == np.array([6,7,8]))
    assert np.all(get_1d_idx(2,np.array([3,5]),6) == np.array([9,11]))
    assert np.all(get_1d_idx(3,np.array([4]),6) == np.array([12]))
    assert np.all(get_1d_idx(4,np.array([5]),6) == np.array([14]))
    # i > j
    assert np.all(get_1d_idx(3,np.array([1,2,4]),6) == np.array([6,9,12]))
    assert np.all(get_1d_idx(4,np.array([3]),6) == np.array([12]))
    assert np.all(get_1d_idx(5,np.array([4]),6) == np.array([14]))