# Module to visualize the performance of machine learning models.
import numpy as np
import matplotlib.pyplot as plt

from matplotlib.axes import Axes
from sklearn.base import ClassifierMixin
from sklearn.utils import check_scalar, check_array


def plot_decision_boundary(clf, bound, res=21, ax=None, cmap='coolwarm_r',
                           boundary_dict=None, confidence_dict=None):
    """Plot the decision boundary of the given classifier.
    Parameters
    ----------
    clf: ClassifierMixin
        The classifier whose decision boundary is plotted.
    bound: array-like, [[xmin, ymin], [xmax, ymax]]
        Determines the area in which the boundary is plotted.
    res: int, optional (default=21)
        The resolution of the plot.
    ax: matplotlib.axes.Axes, optional (default=None)
        The axis on which the boundary is plotted.
    cmap: str | matplotlib.colors.Colormap, optional (default='coolwarm_r')
        The colormap for the confidence levels.
    boundary_dict: dict, optional (default=None)
        Additional parameters for the boundary contour.
    confidence_dict: dict, optional (default=None)
        Additional parameters for the confidence contour. Must not contain a
        colormap because cmap is used.
    """
    # Check input parameters.
    if not isinstance(clf, ClassifierMixin) and not hasattr(clf, 'predict_proba'):
        raise TypeError("`clf` must be a `ClassifierMixin`.")
    check_scalar(res, 'res', int, min_val=1)
    if ax is None:
        ax = plt.gca()
    if not isinstance(ax, Axes):
        raise TypeError("ax must be a matplotlib.axes.Axes.")
    check_array(bound)
    xmin, ymin, xmax, ymax = np.ravel(bound)
    boundary_args = {'colors': 'k', 'linewidths': [2], 'zorder': 1}
    if boundary_dict is not None:
        if not isinstance(boundary_dict, dict):
            raise TypeError("boundary_dict' must be a dictionary.")
        boundary_args.update(boundary_dict)
    confidence_args = {'alpha': 0.9, 'cmap': cmap}
    if confidence_dict is not None:
        if not isinstance(confidence_dict, dict):
            raise TypeError("confidence_dict' must be a dictionary.")
        confidence_args.update(confidence_dict)

    # Create mesh of samples for plotting.
    x_vec = np.linspace(xmin, xmax, res)
    y_vec = np.linspace(ymin, ymax, res)
    X_mesh, Y_mesh = np.meshgrid(x_vec, y_vec)
    mesh_instances = np.array([X_mesh.reshape(-1), Y_mesh.reshape(-1)]).T

    # Predict class-membership probabilities for samples of the mesh.
    posteriors = clf.predict_proba(mesh_instances)
    posteriors = posteriors[:, 0].reshape(X_mesh.shape)

    # Create contour plot of the posterior probabilities.
    ax.contourf(X_mesh, Y_mesh, posteriors, **confidence_args)

    # Plot decision boundary.
    ax.contour(X_mesh, Y_mesh, posteriors, [.5], **boundary_args)
