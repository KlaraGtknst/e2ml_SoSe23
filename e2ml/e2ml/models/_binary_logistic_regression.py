import numpy as np

from e2ml.evaluation import binary_cross_entropy_loss

from scipy.special import expit
from scipy.optimize import minimize

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils import check_array, column_or_1d, check_consistent_length, check_scalar
from sklearn.preprocessing import LabelEncoder


class BinaryLogisticRegression(BaseEstimator, ClassifierMixin):
    """BinaryLogisticRegression

    Binary logistic regression (BLR) is a simple probabilistic classifier for binary classification problems.

    Parameters
    ----------
    maxiter : int, default=100
        Maximum number of optimization steps. (number of iterations)
    lmbda: float, default=0.0
        Regularization hyperparameter.


    Attributes
    ----------
    w_: numpy.ndarray, shape (n_features,)
        Weights (parameters) optimized during training the BLR model.
    """

    def __init__(self, maxiter=100, lmbda=0.0):
        self.y_ = None
        self.maxiter = maxiter
        self.lmbda = lmbda

    def fit(self, X, y):
        """
        Fit the `BinaryLogisticRegression` model using `X` as training data and `y` as class labels.

        Parameters
        ----------
        X: matrix-like, shape (n_samples, n_features)
            The sample matrix `X` is the feature matrix representing the samples for training.
        y: array-like, shape (n_samples) or (n_samples, n_outputs)
            The array `y` contains the class labels of the training samples.

        Returns
        -------
        self: BinaryLogisticRegression,
            The `BinaryLogisticRegression` model fitted on the training data.
        """
        # Check attributes.
        check_scalar(self.maxiter, min_val=0, name='maxiter', target_type=int)
        check_scalar(self.lmbda, min_val=0, name='lmbda', target_type=(int, float))
        X = check_array(X)
        self._check_n_features(X, reset=True)
        y = column_or_1d(y)
        check_consistent_length(X, y)

        # Fit `LabelEncoder` object as `self.label_encoder_`.
        self.label_encoder_ = LabelEncoder()
        self.label_encoder_.fit(y)

        # Raise `ValueError` if there are more than two classes.
        if self.label_encoder_.classes_.shape[0] > 2:
            raise ValueError('There are more than two classes.')

        # Transform `self.y_` using the fitted `self.label_encoder_`.
        self.y_ = self.label_encoder_.transform(y)

        # Initialize weights `w0`.
        w0 = np.zeros((X.shape[1]))

        def loss_func(w):
            """
            Compute the (scaled) loss with respect to weights `w`.

            Parameters
            ----------
            w : np.ndarray of shape (n_features,)

            Returns
            -------
            loss : float
                Evaluated (scaled) loss.
            """
            # Compute predictions for given weights.
            y_pred = expit(np.matmul(X, w))

            # Compute binary cross entropy loss including regularization.
            loss = binary_cross_entropy_loss(y_true=y, y_pred=y_pred)
            loss += 0.5 * len(X)**(-1) * self.lmbda * w.T @ w   # second term

            return loss

        def gradient_func(w):
            # Compute predictions for given weights.
            y_pred = expit(np.matmul(X, w))

            # Compute gradient.
            # y_pred 1 dim vector -> expand dims, or index [:, None] -> 1 dim ndarry to 2 dim ndarray
            # basis function was supplied to X
            gradient = 1 * len(X)**(-1) * np.sum((y_pred - y)[:, None] * X)

            return gradient

        # Use `scipy.optimize.minimize` with `BFGS` as `method` to optimize the loss function and store the result as
        # `self.w_`
        # TODO 

        return self

    def predict_proba(self, X):
        """
        Return probability estimates for the test data `X`.

        Parameters
        ----------
        X:  array-like of shape (n_samples, n_features)
            The sample matrix `X` is the feature matrix representing the training samples.

        Returns
        -------
        P:  numpy.ndarray of shape (n_samples, n_classes)
            The class probabilities of the input samples. Classes are ordered by lexicographic order.
        """
        # Check `X` parameter.
        X = check_array(X)
        self._check_n_features(X, reset=False)  # do we have n features in X?

        # Estimate and return conditional class probabilities.
        y_pred = expit(np.matmul(X, self.w_)) # not a matrix -> ndarray; @ is equal to matmul
        # wrong: P = y_pred.reshape(X.shape[0], X.shape[1])    # column_stack works on tupels, reshape works on ndarray
        P = np.column_stack(1-y_pred, y_pred)   # two dim matrix: entries for both probabilities -> later provide class (higher prob == class)

        return P

    def predict(self, X):
        """
        Return class label predictions for the test data `X`.

        Parameters
        ----------
        X:  array-like of shape (n_samples, n_features)
            Test samples.

        Returns
        -------
        y:  numpy.ndarray of shape = [n_samples]
            Predicted class labels class.
        """
        # Predict class labels `y`.
        proba = self.predict_proba(X)   # matrix with individual class probabilities for 0 and 1
        y = proba.argmax(axis=1)        # return index/class with highest value for every row

        # Re-transform predicted labels using `self.label_encoder_`.
        y = self.label_encoder_.inverse_transform(y)

        return y