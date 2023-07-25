"""
This code provides the function two matched sample t test for the E2ML course WS21 exercise 11.
"""

import numpy as np
from scipy.stats import t
from math import sqrt

__all__ = ['two_matched_sample_t_test', 't_wrapper']

def t_wrapper(p, v):
    """
    Look up the percent point function of the t distribution

    Parameters
    ----------
    p : float
        Percentage to look up (0 <= p <= 100)
    v : float
        Level of significance
    """
    return t.ppf(1-p/100, v)

def two_matched_sample_t_test(xs, ys, alpha=0.05):
    """
    Perform a Two Matched Sample T-test on to list-likes of values

    Parameters
    ----------
    xs,ys : 

    Optional
    --------
    alpha : float
        Level of significance, in percent. (default = 0.05)

    Returns
    -------
    (flag, t, lev) : (bool, float, float)
        flag contains whether the H0 hyptothesis would be accepted, lev is the value returned by looking up in distribution
    """

    # compute values d_i's and d
    # BEGIN SOLUTION

    # ensure xs and ys are numpy arrays
    xs = np.array(xs)
    ys = np.array(ys)
    
    ds = abs(xs-ys)
    d = abs(np.mean(xs)-np.mean(ys))
    # END SOLUTION

    # compute standard deviation
    s = sqrt(sum((ds - d)**2)/(n-1)) # <-- SOLUTION
    
    # compute value t
    if d == 0:
        t = 0 # <-- SOLUTION
    else:
        t = d / (s/sqrt(n)) # <-- SOLUTION

    ## lookup value in t distribution (you can use the t_wrapper function provided above)
    # BEGIN SOLUTION
    lev = t_wrapper(alpha, n-1)
    flag = True if t <= lev else False
    # END SOLUTION

    return (flag, t, lev)
