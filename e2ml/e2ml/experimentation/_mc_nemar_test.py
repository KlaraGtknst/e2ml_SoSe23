"""
This code provides the function McNemar's test for the E2ML course WS21 exercise 11.
"""

import numpy as np
from scipy.stats import chi2

__all__ = ['mc_nemar_test', 'chi2_wrapper']

def chi2_wrapper(p, v):
    """
    Look up the percent point function of the chi-squared distribution

    Parameters
    ----------
    p : float
        Percentage to look up (0 <= p <= 100)
    v : float
        Level of significance
    """
    return chi2.ppf(1-p/100, v)

def mc_nemar_test(y1, y2, alpha=0.05):
    """
    Perform a McNemar test on two list-likes

    Parameters
    ----------
    y1, y2 : np array
        True labels for samples in X_test

    Optional
    --------
    alpha : float
        Level of significance, in percent. (default = 0.05)

    Returns
    -------
    (flag, chi, lev) : (bool, float, float)
        flag contains whether the H0 hyptothesis would be accepted, lev is the value returned by looking up in distribution
    """

    # compute the matrix entries and chi
    # BEGIN SOLUTION
    # ensure y1 and y2 are numpy arrays
    y1 = np.array(y1)
    y2 = np.array(y2)
    
    c00 = sum((y1 != y_test) == (y2 != y_test))
    c01 = sum((y1 != y_test) == (y2 == y_test))
    c10 = sum((y1 == y_test) == (y2 != y_test))
    c11 = sum((y1 == y_test) == (y2 == y_test))
    
    chi = (abs(c01 - c10) - 1)**2 / (c01+c10)
    # END SOLUTION


    # compare to chi tabulated value (you may use chi2_wrapper for that)
    # BEGIN SOLUTION
    lev = chi2_wrapper(alpha, 1)
    flag = True if chi <= lev else False
    # END SOLUTION

    return (flag, chi, lev)
