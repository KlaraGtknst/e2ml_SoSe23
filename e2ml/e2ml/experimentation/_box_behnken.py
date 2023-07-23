"""
This code provides the function box_behnken for the E2ML course WS21 exercise 6.
"""

import numpy as np
from e2ml.experimentation._full_factorial import full_fac

__all__ = ['box_behnken']


# Helping functions

# You may need the 2-level full factoral design
def ff2n(n):  # n is int numer of factors
    return 2 * full_fac([2] * n) - 1


# You may also need repeat center (Creates a 2d-array as center-point portion of a design matrix (elements all zero))
def repeat_center(n, repeat):  # n is int number of factors in original design and repeat stands for int number of
    # center points
    # >>> repeat_center(3, 2) = array([[ 0.,  0.,  0.], [ 0.,  0.,  0.]])
    return np.zeros((repeat, n))


# Implement Box Behnken
def box_behnken(n, center=None):
    """
    Create a Box-Behnken design
    
    Parameters
    ----------
    n : int
        The number of factors in the design
    
    Optional
    --------
    center : int
        The number of center points to include (default = 1).
    
    Returns
    -------
    mat : 2d-array
        The design matrix
    
    Example
    -------
    ::
    
        >>> bbdesign(3)
        array([[-1., -1.,  0.],
               [ 1., -1.,  0.],
               [-1.,  1.,  0.],
               [ 1.,  1.,  0.],
               [-1.,  0., -1.],
               [ 1.,  0., -1.],
               [-1.,  0.,  1.],
               [ 1.,  0.,  1.],
               [ 0., -1., -1.],
               [ 0.,  1., -1.],
               [ 0., -1.,  1.],
               [ 0.,  1.,  1.],
               [ 0.,  0.,  0.],
               [ 0.,  0.,  0.],
               [ 0.,  0.,  0.]])
        
    """

    assert n >= 3  # Number of variables must be at least 3

    # First, compute a factorial DOE with 2 parameters
    mat_fact = ff2n(2)
    # Now populate the real DOE with this DOE

    # Make a factorial design on each pair of dimensions 
    # - So, you created a factorial design with two factors 
    # - Make two loops 

    Index = 0  # <-- SOLUTION
    nb_lines = (n * (n - 1) / 2) * mat_fact.shape[0]  # <-- SOLUTION
    mat = repeat_center(int(n), int(nb_lines))  # <-- SOLUTION

    for i in range(n - 1):  # <-- SOLUTION
        for j in range(i + 1, n):  # <-- SOLUTION
            Index = Index + 1  # <-- SOLUTION
            mat[max([0, (Index - 1) * mat_fact.shape[0]]):Index * mat_fact.shape[0], i] = mat_fact[:, 0]  # <-- SOLUTION
            mat[max([0, (Index - 1) * mat_fact.shape[0]]):Index * mat_fact.shape[0], j] = mat_fact[:, 1]  # <-- SOLUTION

    if center is None:  # <-- SOLUTION
        if n <= 16:  # <-- SOLUTION
            points = [0, 0, 0, 3, 3, 6, 6, 6, 8, 9, 10, 12, 12, 13, 14, 15, 16]  # <-- SOLUTION
            center = points[n]  # <-- SOLUTION
        else:  # <-- SOLUTION
            center = n  # <-- SOLUTION

    mat = np.c_[mat.T, repeat_center(n, center).T].T  # <-- SOLUTION

    return mat
