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
def _majority_label(labels):
    """Return the most common label"""
    labels = np.asarray(labels)
    if labels.ndim != 1 or labels.size == 0:
        raise ValueError("A leaf needs a one-dimensional list of at least one label")
    classes, counts = np.unique(labels, return_counts=True)
    return classes[np.argmax(counts)]
def _check_integer(name, value, minimum):
    """Validate an integer setting"""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer")
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return int(value)
def _check_features(X):
    """Return a numeric array with shape"""
    X = np.asarray(X)
    if X.ndim != 2 or X.shape[1] == 0:
        raise ValueError("X must be a 2-D array with at least one feature column")
    if X.dtype.kind not in "biuf":
        raise TypeError("X must contain numeric or boolean features")
    if np.isinf(X).any():
        raise ValueError("Features may contain NaN, but not infinity")
    return X
def _check_training_data(X, y):
    """Validate a non-empty training chunk with one class label per row."""
    X = _check_features(X)
    y = np.asarray(y)
    if X.shape[0] == 0:
        raise ValueError("Training data must contain at least one row")
    if y.shape != (X.shape[0],):
        raise ValueError("y must be one-dimensional with one label per row")
    if y.dtype.kind not in "biufUS":
        raise TypeError("Class labels must be numbers, booleans or strings")
    if y.dtype.kind == "f" and not np.isfinite(y).all():
        raise ValueError("Class labels must not contain NaN or infinity")
    return X, y
class DecisionTreeClassifier:
    """A Gini decision tree that retrains when its sample buffer fills"""
    def __init__(
        self,
        max_depth=5,
        min_samples_split=2,
        max_features=None,
        buffer_size=500,
        random_state=0,
    ):
        self.max_depth = _check_integer("max_depth", max_depth, 0)
        self.min_samples_split = _check_integer(
            "min_samples_split", min_samples_split, 2
        )
        self.max_features = (
            None if max_features is None
            else _check_integer("max_features", max_features, 1)
        )
        self.buffer_size = _check_integer("buffer_size", buffer_size, 1)
        self.random_state = _check_integer("random_state", random_state, 0)
        self._root = None
        self._X_chunks = []
        self._y_chunks = []
        self.buffered_samples_ = 0
        self.n_retrains_ = 0
        self.n_features_in_ = None
        self.classes_ = None
    def _best_question(self, X, y, rng):
        """Return (feature, threshold, binary) or None if no split helps."""
        columns = np.arange(X.shape[1])
        if self.max_features is not None and self.max_features < columns.size:
            columns = np.sort(
                rng.choice(columns, size=self.max_features, replace=False)
            )
        selected = X[:, columns]
        if np.all((selected == 0) | (selected == 1)):
            local_column, _ = _best_binary_feature(selected, y)
            if local_column is None:
                return None
            return int(columns[local_column]), 1, True
        best_question = None
        best_gain = 0.0
        for column in columns:
            values = X[:, column]
            observed = np.unique(values[~np.isnan(values)])
            if observed.size == 0:
                continue
            # Each distinct value except the largest can separate two
            thresholds = observed[:-1]
            for threshold in thresholds:
                yes_mask = values > threshold
                gain = _score_split(y, yes_mask)
                if gain > best_gain:
                    best_gain = gain
                    best_question = (int(column), threshold, False)
        return best_question
    @staticmethod
    def _yes_mask(X, question):
        column, threshold, binary = question
        if binary:
            return X[:, column] == 1
        return X[:, column] > threshold
    def _build(self, X, y, depth, classes, rng):
        """Build one node, then recursively build its children if needed"""
        labels, counts = np.unique(y, return_counts=True)
        if (
            depth >= self.max_depth
            or y.size < self.min_samples_split
            or labels.size == 1
        ):
            question = None
        else:
            question = self._best_question(X, y, rng)

        if question is None:
            probabilities = np.zeros(classes.size, dtype=float)
            probabilities[np.searchsorted(classes, labels)] = counts / y.size
            return {
                "prediction": _majority_label(y),
                "probabilities": probabilities,
            }
        yes = self._yes_mask(X, question)
        return {
            "question": question,
            "yes": self._build(X[yes], y[yes], depth + 1, classes, rng),
            "no": self._build(X[~yes], y[~yes], depth + 1, classes, rng),
        }
    def _train(self, X, y):
        """Build a replacement tree before installing it"""
        if self.max_features is not None and self.max_features > X.shape[1]:
            raise ValueError("max_features cannot exceed the number of columns")
        classes = np.unique(y)
        rng = np.random.default_rng(self.random_state)
        root = self._build(X, y, depth=0, classes=classes, rng=rng)
        self._root = root
        self.classes_ = classes
        self.n_features_in_ = X.shape[1]
    def _clear_buffer(self):
        self._X_chunks.clear()
        self._y_chunks.clear()
        self.buffered_samples_ = 0
    def fit(self, X, y):
        """Train immediately from scratch and start a fresh streaming buffer"""
        X, y = _check_training_data(X, y)
        self._train(X, y)
        self._clear_buffer()
        self.n_retrains_ = 0
        return self
    def partial_fit(self, X, y):
        """Buffer a chunk and retrain on all buffered rows once the buffer fills"""
        X, y = _check_training_data(X, y)
        if self.n_features_in_ is not None and X.shape[1] != self.n_features_in_:
            raise ValueError("Each chunk must have the same number of feature columns")
        if self.max_features is not None and self.max_features > X.shape[1]:
            raise ValueError("max_features cannot exceed the number of columns")
        new_count = self.buffered_samples_ + X.shape[0]
        if new_count < self.buffer_size:
            self._X_chunks.append(X.copy())
            self._y_chunks.append(y.copy())
            self.buffered_samples_ = new_count
            self.n_features_in_ = X.shape[1]
            return self
        X_train = np.concatenate([*self._X_chunks, X], axis=0)
        y_train = np.concatenate([*self._y_chunks, y], axis=0)
        self._train(X_train, y_train)
        self._clear_buffer()
        self.n_retrains_ += 1
        return self
    def _leaf_groups(self, X):
        """Yield each reached leaf"""
        if self._root is None:
            raise ValueError("Train with fit() or fill the buffer before predicting")
        X = _check_features(X)
        if X.shape[1] != self.n_features_in_:
            raise ValueError("Prediction data must have the same columns as training data")
        pending = [(self._root, np.arange(X.shape[0]))]
        while pending:
            node, rows = pending.pop()
            if rows.size == 0:
                continue
            if "question" not in node:
                yield node, rows
                continue
            yes = self._yes_mask(X[rows], node["question"])
            pending.append((node["yes"], rows[yes]))
            pending.append((node["no"], rows[~yes]))
    def predict(self, X):
        """Return one predicted class label per row, with shape"""
        X = _check_features(X)
        if self.classes_ is None:
            raise ValueError("Train with fit() or fill the buffer before predicting")
        predictions = np.empty(X.shape[0], dtype=self.classes_.dtype)
        for leaf, rows in self._leaf_groups(X):
            predictions[rows] = leaf["prediction"]
        return predictions
    def predict_proba(self, X):
        """Return class probabilities"""
        X = _check_features(X)
        if self.classes_ is None:
            raise ValueError("Train with fit() or fill the buffer before predicting")
        probabilities = np.empty((X.shape[0], self.classes_.size), dtype=float)
        for leaf, rows in self._leaf_groups(X):
            probabilities[rows] = leaf["probabilities"]
        return probabilities