import pytest
import numpy as np

from repliclust.maxmin.utils import sample_with_maxmin

F_CONSTRAINT_SIMPLE = lambda ref_val: (lambda x: 2*ref_val - x)

def test_sample_with_maxmin():
    # Test cases throwing exceptions
    args_causing_exception = [
        # negative vals
        {'n_samples': 10, 'ref_val': -2, 'min_val': 1, 
            'maxmin_ratio': 1.5},
        {'n_samples': 10, 'ref_val': 2, 'min_val': -1, 
            'maxmin_ratio': 1.5},
        {'n_samples': 10, 'ref_val': 2, 'min_val': 1, 
            'maxmin_ratio': -1.5},
        # zeros vals
        {'n_samples': 0, 'ref_val': 2, 'min_val': 1, 
            'maxmin_ratio': 1.5},
        {'n_samples': 10, 'ref_val': 0, 'min_val': 1, 
            'maxmin_ratio': 1.5},
        {'n_samples': 10, 'ref_val': 2, 'min_val': 0, 
            'maxmin_ratio': 1.5},
        {'n_samples': 10, 'ref_val': 2, 'min_val': 1, 
            'maxmin_ratio': 0},
        # ref < min
        {'n_samples': 10, 'ref_val': 1, 'min_val': 2, 
            'maxmin_ratio': 1.5},
        # ref > max
        {'n_samples': 10, 'ref_val': 10, 'min_val': 1, 
            'maxmin_ratio': 1.5},
        # maxmin_ratio < 1
        {'n_samples': 10, 'ref_val': 2, 'min_val': 1, 
            'maxmin_ratio': 0.7},
        # maxmin_ratio = 1, ref != min_val
        {'n_samples': 10, 'ref_val': 2, 'min_val': 1, 
            'maxmin_ratio': 1},
        ]

    with pytest.raises(ValueError):
        for args in args_causing_exception:
            args['f_constraint'] = F_CONSTRAINT_SIMPLE(args['ref_val'])
            sample_with_maxmin(**args)

    # Test cases with appropriate inputs (randomized)
    args_appropriate_input = []
    max_ref_val = 10
    max_maxmin_ratio = 100

    for i in range(100):
        # Step 1: sample reference value
        # Step 2: sample minimum value to be smaller than reference
        # Step 3: sample maxmin ratio to be at least ref/min
        # Step 4: compute max val from min val and maxmin ratio
        ref_val = np.random.uniform(0, max_ref_val)
        maxmin_ratio = np.random.uniform(1, max_maxmin_ratio)
        min_val = 2*ref_val/(1+maxmin_ratio)
        max_val = min_val * maxmin_ratio
   
        args_appropriate_input.append(
            {
            # Do the first 10 tests on the edge case n_samples=1
            'n_samples': (np.random.choice(np.arange(2,15))
                          if i > 10 else 1),
            'min_val': min_val,
            'ref_val': ref_val,
            'maxmin_ratio': maxmin_ratio,
            }
            )

    # Add test case with large sample size
    args_appropriate_input.append({'n_samples': 10000, 'ref_val': 2,
                                    'min_val': 1, 'maxmin_ratio': 3})

    for args in args_appropriate_input:
        args['f_constraint'] = F_CONSTRAINT_SIMPLE(args['ref_val'])
        out = sample_with_maxmin(**args)
        print(args)
        print(out)
        print('min',np.min(out))
        print('max',np.max(out))
        print('ref',args['ref_val'])
        assert check_sample_with_maxmin_output(out, 
                                               args['f_constraint'])


def check_sample_with_maxmin_output(sampled_vals, f_constraint):
    """
    Check that
    - the output of sample_with_maxmin satisfies the lower and
        upper bounds.
    - the minimum and maximum values satisfy the constraint.
    - the output is sorted.
    """
    return (is_sorted(sampled_vals, order='ascending')
                and np.allclose(f_constraint(np.max(sampled_vals)),
                                np.min(sampled_vals))
                and np.allclose(f_constraint(np.min(sampled_vals)),
                                np.max(sampled_vals)))


def is_sorted(values, order='ascending'):
    """ Check that the input is sorted. """
    if order=='ascending':
        return np.all(values[1:] - values[:-1] >= 0)
    elif order=='descending':
        return np.all(values[1:] - values[:-1] <= 0)





