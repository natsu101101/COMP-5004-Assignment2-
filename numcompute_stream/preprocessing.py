"""Streaming preprocessing for arrays shaped (rows, features).
fit(X) starts fresh. partial_fit(X) learns from another chunk.
Numeric transformers calculate statistics separately for each column.
"""
import numpy as np
from numcompute import preprocessing as batch
NotFittedError = batch.NotFittedError
def _numeric_chunk(transformer, X):
    """Check that a numeric chunk matches the earlier chunks."""
    X = batch._check_fit_data(X)
    if hasattr(transformer, "n_features_in_"):
        batch._check_n_features(X, transformer.n_features_in_)
    return X
class _StreamingFit:
    """Give each streaming transformer a fit method that starts fresh."""
    def fit(self, X):
        fresh = type(self)(**self._settings())
        fresh.partial_fit(X)
        self.__dict__.update(fresh.__dict__)
        return self
    def _settings(self):
        return {}
class StandardScaler(_StreamingFit, batch.StandardScaler):
    """Maintain a running mean and population variance for each feature."""
    def partial_fit(self, X):
        X = _numeric_chunk(self, X)
        valid = ~np.isnan(X)
        count = valid.sum(axis=0)
        old_count = getattr(self, "count_", np.zeros_like(count))
        total = old_count + count
        chunk_mean = np.nansum(X / np.maximum(count, 1), axis=0)
        old_mean = getattr(self, "mean_", np.zeros(X.shape[1]))
        old_var = getattr(self, "var_", np.zeros(X.shape[1]))
        share = count / np.maximum(total, 1)
        old_share = old_count / np.maximum(total, 1)
        mean = old_mean * old_share + chunk_mean * share
        centered = np.where(valid, X - chunk_mean, 0.0)
        chunk_var = np.sum(
            (centered / np.sqrt(np.maximum(count, 1))) ** 2,
            axis=0,
        )
        delta = np.where(
            (old_count > 0) & (count > 0),
            chunk_mean - old_mean,
            0.0,
        )
        variance = old_share * old_var + share * chunk_var
        variance += (delta * np.sqrt(old_share * share)) ** 2
        self.count_ = total
        self.mean_ = mean
        self.var_ = variance
        std = np.sqrt(variance)
        self.scale_ = np.where(std > 0, std, 1.0)
        self.n_features_in_ = X.shape[1]
        return self
class MinMaxScaler(_StreamingFit, batch.MinMaxScaler):
    """Remember the lowest and highest observed value in each column."""
    def _settings(self):
        return {"feature_range": self.feature_range}
    def partial_fit(self, X):
        X = _numeric_chunk(self, X)
        low = np.fmin.reduce(X, axis=0)
        high = np.fmax.reduce(X, axis=0)
        self.data_min_ = np.fmin(getattr(self, "data_min_", low), low)
        self.data_max_ = np.fmax(getattr(self, "data_max_", high), high)
        self.data_range_ = self.data_max_ - self.data_min_
        self.n_features_in_ = X.shape[1]
        return self
class SimpleImputer(_StreamingFit, batch.SimpleImputer):
    """Fill missing values with a constant or the running column mean.
    For the mean strategy, use fill_value until a column has an observation.
    """
    def _settings(self):
        return {
            "strategy": self.strategy,
            "fill_value": self.fill_value,
        }
    def partial_fit(self, X):
        X = _numeric_chunk(self, X)
        if self.strategy == "constant":
            self.statistics_ = np.full(X.shape[1], self.fill_value)
        else:
            count = np.sum(~np.isnan(X), axis=0)
            old_count = getattr(self, "count_", np.zeros_like(count))
            total = old_count + count
            divisor = np.maximum(total, 1)
            old_mean = getattr(self, "statistics_", np.zeros(X.shape[1]))
            mean = old_mean * (old_count / divisor)
            mean += np.nansum(X / divisor, axis=0)
            self.statistics_ = np.where(total > 0, mean, self.fill_value)
            self.count_ = total
        self.n_features_in_ = X.shape[1]
        return self
Imputer = SimpleImputer
class OneHotEncoder(_StreamingFit, batch.OneHotEncoder):
    """Add columns for new categories while keeping existing column positions."""
    def _settings(self):
        return {"handle_unknown": self.handle_unknown}
    def partial_fit(self, X):
        X, family = self._check_categorical_2d(X)
        if X.shape[0] == 0:
            raise ValueError("A training chunk must contain at least one row")
        if hasattr(self, "n_features_in_"):
            batch._check_n_features(X, self.n_features_in_)
            if family != self._family:
                raise TypeError("Category types must be consistent across chunks")
        else:
            self._family = family
            self.n_features_in_ = X.shape[1]
            self.n_features_out_ = 0
            self._positions = [{} for _ in range(X.shape[1])]
        for column, positions in enumerate(self._positions):
            for value in np.unique(X[:, column]).tolist():
                if value not in positions:
                    positions[value] = self.n_features_out_
                    self.n_features_out_ += 1
        self.categories_ = [
            np.asarray(list(positions)) for positions in self._positions
        ]
        return self
    def transform(self, X):
        """Return one row of indicator values for each input row."""
        self._check_is_fitted()
        X, family = self._check_categorical_2d(X)
        batch._check_n_features(X, self.n_features_in_)
        if family != self._family:
            raise TypeError("Category types must match the training data")
        output = np.zeros((X.shape[0], self.n_features_out_))
        rows = np.arange(X.shape[0])
        for column, positions in enumerate(self._positions):
            values, inverse = np.unique(X[:, column], return_inverse=True)
            indices = np.array(
                [positions.get(value, -1) for value in values.tolist()],
                dtype=int,
            )
            if self.handle_unknown == "error" and np.any(indices < 0):
                raise ValueError(f"Unseen category in feature {column}")
            target = indices[inverse.reshape(-1)]
            known = target >= 0
            output[rows[known], target[known]] = 1.0
        return output