import pytest
import numpy as np
from test_gradients_marginal_lda import get_1d_idx

def test_get_1d_idx():
    assert get_1d_idx(0,np.array([1,2,3,4]),6) == np.array([0,1,2,3])
    assert get_1d_idx(1,np.array([3,4,5]),6) == np.array([6,7,8])
    assert get_1d_idx(2,np.array([3,5]),6) == np.array([8,10])