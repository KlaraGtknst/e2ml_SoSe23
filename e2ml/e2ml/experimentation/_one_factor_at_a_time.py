import numpy as np
from sklearn.utils import column_or_1d, check_scalar


def one_factor_at_a_time(levels):
    """
    Implements the standard one-factor-at-a-time approach for given factor-wise levels.

    Parameters
    ----------
    levels : array-like of shape (n_factors,)
        Integer array indicating the number of levels of each input design factor (variable).

    Returns
    -------
    X : np.ndarray of shape (n_combs, n_factors)
        Design matrix with coded levels 0 to k-1 for a k-level factor, each one at a time.
    """
    levels = np.array(levels)
    X = np.zeros((levels.sum() + 1 - len(levels), len(levels)))
    summed_levels = 1

    for l, level in enumerate(levels):
        for j, lev in enumerate(range(1, level)):
            X[summed_levels+j, l] = lev

        summed_levels += level-1

    return X.astype(int)
