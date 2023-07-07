import numpy as np
import itertools

from sklearn.utils.validation import check_array
from scipy import stats

from ._one_sample_tests import t_test_one_sample


def t_test_paired(sample_data_1, sample_data_2=None, mu_0=0, test_type="two-sided"):
    """Perform a paired t-test.

    Parameters
    ----------
    sample_data_1 : array-like of shape (n_samples,)
        Sample data drawn from a population 1. If no sample data is given, `sample_data_1` is assumed to consist of
        differences.
    sample_data_2 : array-like of shape (n_samples,), optional (default=None)
        Sample data drawn from a population 2.
    mu_0 : float or int
        Population mean assumed by the null hypothesis.
    test_type : {'right-tail', 'left-tail', 'two-sided'}
        Specifies the type of test for computing the p-value.

    Returns
    -------
    t_statistic : float
        Observed t-transformed test statistic.
    p : float
        p-value for the observed sample data.
    """
    # Check parameters.
    sample_data_1 = check_array(sample_data_1, ensure_2d=False)
    if sample_data_2 is not None:
        sample_data_2 = check_array(sample_data_2, ensure_2d=False)

    # TODO
    # test statistic: mean of difference of sample, expected value: 0, bc they ought to be equally distributed
    sample_data_diff = sample_data_1 if sample_data_2 is None else sample_data_1 - sample_data_2
    return t_test_one_sample(sample_data_diff, mu_0=mu_0, test_type=test_type)


def wilcoxon_signed_rank_test(sample_data_1, sample_data_2=None, test_type="two-sided"):
    """Perform a Wilcoxon signed-rank test.

    Parameters
    ----------
    sample_data_1 : array-like of shape (n_samples,)
        Sample data drawn from a population 1. If no sample data is given, `sample_data_1` is assumed to consist of
        differences.
    sample_data_2 : array-like of shape (n_samples,), optional (default=None)
        Sample data drawn from a population 2.
    test_type : {'right-tail', 'left-tail', 'two-sided'}
        Specifies the type of test for computing the p-value.

    Returns
    -------
    w_statistic : float
        Observed positive rank sum as test statistic.
    p : float
        p-value for the observed sample data.
    """
    # Check parameters.
    sample_data_1 = check_array(sample_data_1, ensure_2d=False)
    if sample_data_2 is not None:
        sample_data_2 = check_array(sample_data_2, ensure_2d=False)
    if test_type not in ["two-sided", "left-tail", "right-tail"]:
        raise ValueError("`test_type` must be in `['two-sided', 'left-tail', 'right-tail']`")

    # TODO 
