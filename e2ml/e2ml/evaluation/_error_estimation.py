import numpy as np

from sklearn.utils.validation import check_scalar, column_or_1d, check_random_state, check_consistent_length


def cross_validation(sample_indices, n_folds=5, random_state=None, y=None):
    """
    Performs a (stratified) k-fold cross-validation.

    Parameters
    ----------
    sample_indices : int
        Array of sample indices.
    n_folds : int, default=5
        Number of folds. Must be at least 2.
    random_state : int, RandomState instance or None, default=None
        `random_state` affects the ordering of the indices, which controls the randomness of each fold.

    Returns
    -------
    train : list
        Contains the training indices of each iteration, where train[i] represents iteration i.
    test : list
        Contains the test indices of each iteration, where test[i] represents iteration i.
    """
    # TODO 

