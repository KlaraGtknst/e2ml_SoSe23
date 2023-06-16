import numpy as np
import matplotlib.pyplot as plt

def roc_curve(labels, x, scores):
    """
    Generate the Receiver Operating Characteristic (ROC) curve for a binary classification problem.

    Parameters
    ----------
    labels : array-like of shape (n_samples,)
        True class labels for each sample.
    x : int or str
        Positive class or class of interest.
    scores : array-like of shape (n_samples,)
        Scores or probabilities assigned to each sample.

    Returns
    -------
    roc_curve : ndarray of shape (n_thresholds, 2)
        Array containing the true positive rate (TPR) and false positive rate (FPR) pairs
        at different classification thresholds.

    """
    x = 1 - x
    labels = np.array(labels)
    scores = np.array(scores)

    # tp, fp ermitteln
    positive_idx = scores[labels == x]
    negative_idx = scores[labels != x]

    # sortieren nach labels (erste negative, dann positive), dann nach scores (klein nach groß)
    #np.lexsort(np.vstack((scores, labels)))
    # iterieren über thresholds
    zs_idx = np.argsort(scores)
    zs = scores[zs_idx] # sorted scores
    ls = labels[zs_idx] # sorted labels

    roc_curve = np.zeros((len(labels), 2))
    for i,t in enumerate(zs):
        pred = (zs >= t) # predicted labels
        tp = (ls == pred) & (ls == x) # true positives: pred==ground truth==positive class
        fp = (ls != pred) & (ls != x) # false positives: pred!=ground truth==positive class FALSCH

        tpr = sum(tp) / sum(ls == x) # true positive rate
        fpr = sum(fp) / sum(ls != x) # false positive rate

        roc_curve[i,:] = [tpr, fpr]

    roc_curve[-1,:] = [1,1]
    return roc_curve


def roc_auc(points):
    """
    Compute the Area Under the Receiver Operating Characteristic (ROC) curve.

    Parameters
    ----------
    points : array-like
        List of points representing the ROC curve. (FPR und TPR an verschiedenen Thresholds)

    Returns
    -------
    auc : float
        Area Under the ROC curve.

    """
    points = np.array(points)
    tmp = np.vstack(([[0,0]], points[np.lexsort(points.T)], [[1,1]]))
    x_dif = tmp[1:,0] - tmp[:-1,0]
    y_dif = tmp[1:,1] + tmp[:-1,1]
    area = np.sum(x_dif * y_dif/2)

    return area



def draw_lift_chart(true_labels, pos, predicted):
    """
    Draw a Lift Chart based on the true labels, positive class, and predicted class labels.

    Parameters
    ----------
    true_labels : array-like
        True class labels for each sample.
    pos : int or str
        Positive class or class of interest.
    predicted : array-like
        Predicted class labels for each sample.

    Returns
    -------
    None

    """
    # thresholds sind egal
    # x-Achse: Anteil der Datenpunkte
    # y-Achse: Anteil der positiven Klasse
    plt.figure(figsize=(10, 6))
    plt.title('Lift Chart')
    plt.xlabel('Sample size')
    plt.ylabel('True positives')
    plt.grid(True)

    # Code aus Übung
    tp = np.ones((len(true_labels), 2))
    tp[:,0] = np.cumsum(tp[:,0])
    for (l,p,i) in zip(true_labels, predicted, np.arange(0, len(true_labels))):
        tp[i,1] = (l == p) & (l == pos)
    tp[:,1] = np.cumsum(tp[:,1])
    plt.plot(tp[:,0], tp[:,1], 'b-')
    plt.scatter(tp[:,0], tp[:,1])
    plt.show()


    # Mareks Code
    n = len(true_labels)    # number of samples

    # store sample index and true labels
    tmp = np.zeros((n, 2))
    tmp[:,0] = np.arange(1,n+1)
    tmp[:,1] = np.array(true_labels)

    with np.nditer(tmp[:,1], op_flags=['readwrite'], flags=['f_index']) as it:
        for x in it:
            if x == pos and x[...] == predicted[it.index]:
                x[...] = 1  # tp
            else:
                x[...] = 0  # kein tp

    tmp[:,1] = np.cumsum(tmp[:,1]) # kumulierte summe der true labels

    plt.scatter(tmp[:,0], tmp[:,1], s=5, c='b', marker='o')
    plt.show()


def kl_divergence(p, q):
    """
    Compute the Kullback-Leibler (KL) divergence between two probability distributions `p` and `q`.

    Parameters
    ----------
    p : array-like
        Probability distribution P.
    q : array-like
        Probability distribution Q.

    Returns
    -------
    kl_div : float
        KL divergence between P and Q.

    """
    p = np.array(p)
    q = np.array(q)

    log_ratio = np.log(p/q)
    kl_div = np.sum(p * log_ratio)

    return kl_div