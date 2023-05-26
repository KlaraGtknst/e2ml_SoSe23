"""
This code provides the implementation for the latin hypercube. 
"""

import numpy as np

from sklearn.utils import check_scalar, check_array


def lat_hyp_cube_unit(n_samples, n_dimensions):

    """
    Generate a latin-hypercube design

    Parameters
    ----------
    n_samples : int
       Number of samples to be generated.

    n_dimensions : int
       Dimensionality of the generated samples.

    Returns
    -------
    X : np.ndarray of shape (n_samples, n_dimensions)
        An `n_samples-by-n_dimensions` design matrix whose levels are spaced between zero and one.
    """
    # TODO 


def lat_hyp_cube(n_samples, n_dimensions, bounds=None):
    """Generate a specified number of samples according to a Latin hypercube in a user-specified bounds.

    Parameters
    ----------
    n_samples : int
       Number of samples to be generated.
    n_dimensions : int
       Dimensionality of the generated samples.
    bounds : None or array-like of shape (n_dimensions, 2)
       `bounds[d, 0]` is the minimum and `bounds[d, 1]` the maximum
       value for dimension `d`.

    Returns
    -------
    X : numpy.ndarray of shape (n_samples, n_dimensions)
       Generated samples.
    """
    # Check parameters.
    check_scalar(n_samples, name="n_samples", target_type=int, min_val=1)
    check_scalar(n_dimensions, name="n_dimensions", target_type=int, min_val=1)
    if bounds is not None:
        bounds = check_array(bounds)
        if bounds.shape[0] != n_dimensions or bounds.shape[1] != 2:
            raise ValueError("`bounds` must have shape `(n_dimensions, 2)`.")
    else:
        bounds = np.zeros((n_dimensions, 2))
        bounds[:, 1] = 1

    # TODO 
