"""
This code provides the function wilcoxon signed rank test for the E2ML course WS21 exercise 11.
"""

import math
import numpy as np
from scipy.stats import norm

__all__ = ['wilcox_signed_rk_test', 'wilcoxon_wrapper', 'normal_wrapper']

wilcox_table = {5:   [0,2,3,5,8,10,13,17,21,25,30,35,41,47,53,60,67,75,83,91,100],
                2.5: [0,0,2,3,5,8,10,13,17,21,25,29,34,40,46,52,58,65,73,81,89],
                1:   [0,0,0,1,3,5,7,9,12,15,19,23,27,32,37,43,49,55,62,69,76],
                0.5: [0,0,0,0,1,3,5,7,9,12,15,19,23,27,32,37,42,48,54,61,68],
                0.1: [0,0,0,0,0,0,1,2,4,6,8,11,14,18,21,26,30,35,40,45,51]}

def wilcoxon_wrapper(p, n):
    """
    Look up the percent point function of the Wilcoxon Signed-Rank distribution.

    Parameters
    ----------
    p : float
        Percentage to look up (0 <= p <= 100)
    n : int
        Number of degrees of freedom
    """
    tmp = wilcox_table.get(p, -1)

    if tmp == -1 or not (5 <= n and n <= 25):
        raise ValueError("Value outside of look-up table for Wilcoxon Signed-Rank distribution")
    
    return tmp[n-5]

def normal_wrapper(p):
    """
    Look up the percent point function of the normal distribution.

    Parameters
    ----------
    p : float
        Percentage to look up (0 <= p <= 100)
    """
    return norm.ppf(1-p/100)


def wilcox_signed_rk_test(xs, ys, alpha=5, split_zeros=True):
    """
    Perform a Wilcox Signed Rank test on two list-likes with
    H0 being that the difference between the (paired) values
    is not statistically significant.

    Parameters
    ----------
    xs, ys : list-likes

    Optional
    --------
    alpha : float
        Level of significance, in percent. (default = 5)
    split_zeros: boolean
        Whether zeros should be added to W_s1 and W_s2 or just ignored

    Returns
    -------
    (flag, T, lev) : (bool, float, float)
        flag contains whether the H0 hyptothesis would be accepted, lev is the value returned
        by looking up in distribution, whereas T is the computed statistic.
    """
    # compute d_i's
    # BEGIN SOLUTION
    d = xs-ys
    d_abs = abs(d)
    # END SOLUTION
    
    # perform the ranking with removal of zeros
    # BEGIN SOLUTION
    d  = d[np.argsort(d_abs)]
    r  = sum(d == 0)
    
    d  = d[d != 0]
    n  = len(d)
    rk = np.arange(n)+1
    for t in np.unique(d):
        idx = d==t
        rk[idx] = np.mean(rk[idx])
    # END SOLUTION
    
    
    # compute W_s1 and W_s2
    # BEGIN SOLUTION
    W_s1 = sum(rk[d>0])
    W_s2 = sum(rk[d<0])
    # END SOLUTION
    
    # add zeros, depending on split_zeros flag
    # BEGIN SOLUTION
    if split_zeros:
        W_s1 += r // 2
        W_s2 += r // 2
        n -= r - 2*(r //2)
    # END SOLUTION
    
    # compute test statistic
    T = max(min(W_s1, W_s2), 0) # <-- SOLUTION
    
    # compare with wilcoxon signed-rank (respectively normal) distribution
    # BEGIN SOLUTION
    if 5 <= n and n <= 25 and alpha in [5,2.5,1,0.5,0.1]:
        lev = wilcoxon_wrapper(alpha, n)
    else:
        lev = normal_wrapper(alpha)
        lev = (n*(n+1)/4) - lev*sqrt((n*(n+1)*(2*n+1))/24)
    # END SOLUTION
    
    flag = True if T <= lev else False
    
    return (flag, T, lev)
