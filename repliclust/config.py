"""
Share a centralized random number generator across program files
for reproducibility.
"""

import numpy as np
global _rng

def init_rng():
    global _rng
    _rng = np.random.default_rng()