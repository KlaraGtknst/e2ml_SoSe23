import numpy as np

from sklearn.utils import check_scalar, check_array


def van_der_corput_sequence(n_max, base=2):
    """Generate van der Corput sequence for 1 <= n <= n_max and given base.

    Parameters
    ----------
    n_max : int
        Number of elements of the sequence.
    base : int
        Base of the sequence.

    Returns
    -------
    sequence : numpy.ndarray of shape (n_max,)
        Generate van der Corput sequence for 1 <= n <= n_max and given base.
    """
    # Check parameters.
    check_scalar(n_max, name="n_max", target_type=int, min_val=1)
    check_scalar(base, name="base", target_type=int, min_val=2)

    sequence = []
    for i in range(1, n_max+1):
        n_th_number, denom = 0.0, 1.0
        while i > 0:
            i, remainder = divmod(i, base)
            denom *= base
            n_th_number += remainder / denom
        sequence.append(n_th_number)
    return np.array(sequence)


def primes_from_2_to(n_max):
    """Generate prime numbers from 2 to n_max.

    Parameters
    ----------
    n_max : int
        Maximum prime number to be generated.

    Returns
    -------
    prime_numbers : numpy.ndarray of shape (n_prime_numbers)
        Array of all prime numbers from 2 to n_max.
    """
    # Check parameters.
    check_scalar(n_max, name="n_max", target_type=int, min_val=2)

    primes = list(range(2, n_max+1))
    for i in range(2, n_max+1):
        for j in range(2, i):
            if i % j == 0:
                primes.remove(i)
                break
    return np.array(primes)



def halton_unit(n_samples, n_dimensions):
    """Generate a specified number of samples according to a Halton sequence in the unit hypercube.

    Parameters
    ----------
    n_samples : int
        Number of samples to be generated.
    n_dimensions : int
        Dimensionality of the generated samples.

    Returns
    -------
    X : numpy.ndarray of shape (n_samples, n_dimensions)
        Generated samples.
    """
    # Check parameters.
    check_scalar(n_samples, name="n_samples", target_type=int, min_val=1)
    check_scalar(n_dimensions, name="n_dimensions", target_type=int, min_val=1)

    big_number = 10
    while "Not enough prime numbers":
        bases = primes_from_2_to(big_number)[:n_dimensions]
        if len(bases) == n_dimensions:
            break
        big_number += 10
    X = [van_der_corput_sequence(n_samples, int(base)) for base in bases]
    X = np.stack(X, axis=-1)

    return X

def halton(n_samples, n_dimensions, bounds=None):
    """Generate a specified number of samples according to a Halton sequence in a user-specified hypercube.

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

    X = halton_unit(n_samples, n_dimensions)

    x_min = bounds[:, 0]
    x_max = bounds[:, 1]
    X = X * (x_max - x_min) + x_min
    return X
