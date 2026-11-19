__all__ = ["Pipeline"]

class GroupPipeline:
    """
    Input:
        steps: list of (name, object) pairs

    Attributes:
        steps: list of (name, object) pairs
        named_steps: dict, name -> object

    Error Raised:
        ValueError: if steps is empty
        TypeError: if a step that is not the last one has no transform method
        (any other missing method gives python's own AttributeError)
    """
    def __init__(self, steps):

        # Validating input steps during initialzation
        if len(steps) == 0:
            raise ValueError("steps must not be empty")
        for name, step in steps[:-1]:
            if not hasattr(step, "transform"):
                raise TypeError(f"step {name!r} is not the last step, so it needs a transform method")

        self.steps = list(steps) # Better to create a copy
        self.named_steps = dict(steps)

    def fit(self, X, y=None):

        if y is not None and len(X) != len(y):
            raise ValueError(f"X and y must have the same number of rows, got {len(X)} and {len(y)}")

        data = X
        for name, step in self.steps[:-1]:
            step.fit(data)
            data = step.transform(data)

        # Getting the last step , usually predict
        last = self.steps[-1][1]
        if hasattr(last, "predict") and y is not None: #as we dont have a model yet
            last.fit(data, y)  # a model learns from y(answers)
        else:
            last.fit(data) # just fits if there is no model
        return self

    def transform(self, X):
        data = X
        for name, step in self.steps:
            data = step.transform(data)
        return data

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)

    def predict(self, X):
        data = X
        for name, step in self.steps[:-1]: #run every step except last , which is usually model training
            data = step.transform(data)
        return self.steps[-1][1].predict(data)

from copy import deepcopy

import numpy as np


class Pipeline(GroupPipeline):
    """Extend the original pipeline with buffered streaming training."""

    def __init__(self, steps):
        super().__init__(steps)
        self._templates = deepcopy(self.steps)
        self._pending = []
        self.buffered_samples_ = 0
        self.n_retrains_ = 0
        self.n_features_in_ = None
        self.is_fitted_ = False

    @property
    def model(self):
        return self.steps[-1][1]

    def fit(self, X, y=None):
        """Train immediately using the original group pipeline."""
        fresh = GroupPipeline(deepcopy(self._templates))
        fresh.fit(X, y)
        self.steps = fresh.steps
        self.named_steps = fresh.named_steps
        self._pending.clear()
        self.buffered_samples_ = 0
        self.n_retrains_ = 0
        self.n_features_in_ = np.asarray(X).shape[1]
        self.is_fitted_ = True
        return self

    def partial_fit(self, X, y):
        """Collect raw rows and retrain when the buffer reaches its limit."""
        X = np.asarray(X)
        y = np.asarray(y)

        if X.ndim != 2 or X.shape[0] == 0 or X.shape[1] == 0:
            raise ValueError("X must be a non-empty 2-D array")
        if y.shape != (X.shape[0],):
            raise ValueError("Provide one label per row")
        if self.n_features_in_ is not None and X.shape[1] != self.n_features_in_:
            raise ValueError("The number of input features changed")

        total = self.buffered_samples_ + X.shape[0]
        if total < self.model.buffer_size:
            self._pending.append((X.copy(), y.copy()))
            self.buffered_samples_ = total
            self.n_features_in_ = X.shape[1]
            return self

        chunks = self._pending + [(X, y)]
        inputs = [features for features, _ in chunks]
        steps = deepcopy(self._templates)

        # Learn one consistent transformation for all rows in this buffer.
        for _, transformer in steps[:-1]:
            for features in inputs:
                transformer.partial_fit(features)
            inputs = [transformer.transform(features) for features in inputs]

        model = steps[-1][1]
        for features, (_, labels) in zip(inputs, chunks):
            model.partial_fit(features, labels)

        self.steps = steps
        self.named_steps = dict(steps)
        self._pending.clear()
        self.buffered_samples_ = 0
        self.n_features_in_ = X.shape[1]
        self.n_retrains_ += 1
        self.is_fitted_ = True
        return self

    def predict(self, X):
        if not self.is_fitted_:
            raise ValueError("Fill the first buffer or call fit() before predicting")
        return super().predict(X)

    def predict_proba(self, X):
        """Return class probabilities using the currently trained pipeline."""
        if not self.is_fitted_:
            raise ValueError("Fill the first buffer or call fit() before predicting")

        data = X
        for _, transformer in self.steps[:-1]:
            data = transformer.transform(data)
        return self.model.predict_proba(data)
