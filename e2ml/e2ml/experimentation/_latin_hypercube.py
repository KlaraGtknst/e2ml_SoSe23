"""
This code provides the implementation for the latin hypercube. 
"""

import numpy as np
import random

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
    # grid of size n_samples = 200 OR just chose 200 random grids
    positions = np.linspace(0, 1, n_samples, endpoint=False, dtype=float)
    grid = np.array(np.meshgrid(*[positions for level in range(n_dimensions)])).T.reshape(-1, n_dimensions)

    # sequence of 200x 2-dim values
    sequence = []

    # choose random grids
    grids_selected = random.choices(grid, k=n_samples)

    # generate random number within interval
    interval_size = float(1/n_samples)
    for i in range(n_samples):
        sample = []
        for dim in range(n_dimensions):
            rand_grid_val = random.uniform(grids_selected[i][dim], grids_selected[i][dim] + interval_size)
            sample.append(rand_grid_val)
        sequence.append(sample)

    return np.array(sequence)


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

    X = lat_hyp_cube_unit(n_samples, n_dimensions)

    x_min = bounds[:, 0]
    x_max = bounds[:, 1]
    X = X * (x_max - x_min) + x_min

    return X
