import numpy as np
def _gini(labels):
    """Calculate Gini impurity"""
    labels = np.asarray(labels)
    if labels.ndim != 1:
        raise ValueError("labels must be one-dimensional")
    if labels.size == 0:
        return 0.0
    _, counts = np.unique(labels, return_counts=True)
    proportions = counts / labels.size
    return float(1.0 - np.sum(proportions ** 2))
def _information_gain(parent_labels, left_labels, right_labels):
    """Calculate the drop in Gini impurity after a split"""
    n_left = len(left_labels)
    n_right = len(right_labels)
    if n_left == 0 or n_right == 0:
        return 0.0
    total = n_left + n_right
    impurity_after = (
        (n_left / total) * _gini(left_labels)
        + (n_right / total) * _gini(right_labels)
    )
    return _gini(parent_labels) - impurity_after