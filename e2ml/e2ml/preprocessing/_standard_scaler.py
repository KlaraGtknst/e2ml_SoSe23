import numpy as np

from sklearn.base import BaseEstimator


class StandardScaler(BaseEstimator):
    """StandardScaler

    Standardize features by removing the mean and scaling to unit variance.

    Attributes
    ----------
    mu_ : numpy.ndarray, shape (n_features)
        The mean value for each feature in the training set.
    sigma_ : numpy.ndarray, shape (n_features)
        The standard deviation for each feature in the training set.
    """

    def fit(self, X):
        """
        Determine required parameters to standardize data.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Input samples.

        Returns
        -------
        self : StandardScaler
            The fitted StandardScaler object.
        """
        # Transform to numpy.ndarray.
        X = np.array(X)

        # Compute `self.mu_` containing the mean value for each feature in the training set.
        self.mu_ = np.mean(X, axis=0)  # <-- SOLUTION

        # Compute `self.mu_` containing the standard deviations for each feature in the training set.
        self.sigma_ = np.std(X, axis=0)  # <-- SOLUTION

        return self

    def transform(self, X):
        """
        Standardizes input samples `X`.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Input samples.

        Returns
        -------
        Z : numpy.ndarray, shape (n_samples, n_features)
            Standardized samples.
        """
        # Transform to numpy.ndarray.
        X = np.array(X)

        # Standardize data by computing `Z`.
        # BEGIN SOLUTION
        with np.errstate(all='ignore'):
            Z = (X - self.mu_) / self.sigma_
            Z = np.nan_to_num(Z, nan=0)
        # END SOLUTION

        return Z

    def inverse_transform(self, Z):
        """
        Scales back the data to the original data representation.

        Parameters
        ----------
        Z : array-like, shape (n_samples, n_features)
            Standardized samples.

        Returns
        -------
        X : numpy.ndarray, shape (n_samples, n_features)
            Re-scaled samples.
        """
        # Transform to numpy.ndarray.
        Z = np.array(Z)

        # Re-scale samples to original space by computing `X`.
        X = Z * self.sigma_ + self.mu_  # <-- SOLUTION

        return X