import numpy as np

from sklearn.utils.validation import check_array, check_scalar
from scipy import stats


def z_test_one_sample(sample_data, mu_0, sigma, test_type="two-sided"):
    """Perform a one-sample z-test.

    Parameters
    ----------
    sample_data : array-like of shape (n_samples,)
        Sample data drawn from a population.
    mu_0 : float or int
        Population mean assumed by the null hypothesis.
    sigma: float
        True population standard deviation.
    test_type : {'right-tail', 'left-tail', 'two-sided'}
        Specifies the type of test for computing the p-value.
        left-tail: Intergral links bis pvalue
        right-tail: 1 - Integral links bis pvalue
        two-sided: min etc.

    Returns
    -------
    z_statistic : float
        Observed z-transformed test statistic.
    p : float
        p-value for the observed sample data.
    """
    # Check parameters.
    sample_data = check_array(sample_data, ensure_2d=False)
    mu_0 = check_scalar(mu_0, name="mu_0", target_type=(int, float))
    sigma = check_scalar(sigma, name="sigma", target_type=(int, float), min_val=0, include_boundaries="neither")
    if test_type not in ["two-sided", "left-tail", "right-tail"]:
        raise ValueError("`test_type` must be in `['two-sided', 'left-tail', 'right-tail']`")

    # empirical mean
    empirical_mean = np.mean(sample_data)

    # sample size
    sample_size = len(sample_data)

    # z_statistic
    # kommen empirical means aus gleichen Distributions?
    z_statistic = (empirical_mean - mu_0) / (sigma / np.sqrt(sample_size))

    # p-value
    # depends on test_type
    p_left = stats.norm.cdf(z_statistic)
    p_right = 1 - p_left
    if test_type == "left-tail":
        p = p_left
    elif test_type == "right-tail":
        p = p_right
    else:
        p = 2 * min(p_left, p_right)

    return z_statistic, p


def t_test_one_sample(sample_data, mu_0, test_type="two-sided"):
    """Perform a one-sample t-test.

    Parameters
    ----------
    sample_data : array-like of shape (n_samples,)
        Sample data drawn from a population.
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

    Variance is estimated from the sample data (not given like in z-test).
    """
    # Check parameters.
    sample_data = check_array(sample_data, ensure_2d=False)
    mu_0 = check_scalar(mu_0, name="mu_0", target_type=(int, float))
    if test_type not in ["two-sided", "left-tail", "right-tail"]:
        raise ValueError("`test_type` must be in `['two-sided', 'left-tail', 'right-tail']`")

    # empirical mean is test statistic
    empirical_mean = np.mean(sample_data)

    # empirical standard deviation
    # ddof=1: Delta degrees of freedom. The divisor used in calculations is N - ddof, where N represents the number of elements.-> N-1 = ddf
    empirical_sigma = np.std(sample_data, ddof=1)

    # sample size
    sample_size = len(sample_data)

    # t_statistic
    # kommen empirical means aus gleichen Distributions?
    # sigma is not given, has to be estimated from sample data
    # degrees of freedom = sample_size - 1; Letzter Datenpunkt ist aus N-1 anderen und mean berechenbar
    t_statistic = (empirical_mean - mu_0) / (empirical_sigma / np.sqrt(sample_size))

    # p-value
    # depends on test_type
    p_left = stats.t.cdf(t_statistic, df=sample_size - 1)    # df = degrees of freedom
    p_right = 1 - p_left
    if test_type == "left-tail":
        p = p_left
    elif test_type == "right-tail":
        p = p_right
    else:
        p = 2 * min(p_left, p_right)

    return t_statistic, p

