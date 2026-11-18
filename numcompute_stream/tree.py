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
def _score_split(labels, yes_mask):
    """Score a yes/no split"""
    labels = np.asarray(labels)
    yes_mask = np.asarray(yes_mask)
    if labels.ndim != 1 or yes_mask.shape != labels.shape:
        raise ValueError("Each label needs one yes/no answer")
    if yes_mask.dtype != np.bool_:
        raise TypeError("The yes/no answers must be True or False")
    yes_labels = labels[yes_mask]
    no_labels = labels[~yes_mask]
    return _information_gain(labels, yes_labels, no_labels)
def _best_binary_feature(X, labels):
    """Return the column with the highest positive gain"""
    X = np.asarray(X)
    labels = np.asarray(labels)
    if X.ndim != 2:
        raise ValueError("X must have one row per student and one column per question")
    if labels.shape != (X.shape[0],):
        raise ValueError("Each row needs one label")
    if not np.all((X == 0) | (X == 1)):
        raise ValueError("Binary features must contain only 0 or 1")
    best_feature = None
    best_gain = 0.0
    for column in range(X.shape[1]):
        yes_mask = X[:, column] == 1
        gain = _score_split(labels, yes_mask)
        if gain > best_gain:
            best_feature = column
            best_gain = gain
    return best_feature, best_gain