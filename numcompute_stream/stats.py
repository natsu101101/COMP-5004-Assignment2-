"""
numcompute.stats

Descriptive statistics and a streaming mean/variance, numpy only.

- data must be numbers, inf is not allowed, empty input raises ValueError
- nan_policy: 'propagate' (NaN in -> NaN out), 'omit' (skip NaN) or 'raise'
- axis=None reduces everything, keepdims=True keeps reduced axes as length 1
"""
import warnings

import numpy as np

__all__ = ['mean', 'median', 'var', 'variance', 'std', 'min', 'max',
           'histogram', 'quantile', 'Welford', 'StreamingStats']


def _check_data(data, nan_policy):
    # check the input and return it as a float array
    if nan_policy not in ('propagate', 'omit', 'raise'):
        raise ValueError("nan_policy must be 'propagate', 'omit' or 'raise'")
    arr = np.asarray(data)
    if arr.dtype.kind not in 'biuf':
        raise TypeError('data must contain real numbers')
    arr = arr.astype(float)
    if arr.size == 0:
        raise ValueError('data must not be empty')
    if np.isinf(arr).any():
        raise ValueError('data must not contain infinity')
    if nan_policy == 'raise' and np.isnan(arr).any():
        raise ValueError('data contains NaN')
    return arr


def _check_ddof(ddof):
    # True/False count as ints in python, so check for bool first
    if isinstance(ddof, (bool, np.bool_)) or not isinstance(ddof, (int, np.integer)):
        raise TypeError('ddof must be a non-negative integer')
    if ddof < 0:
        raise ValueError('ddof must be non-negative')


def _apply(name, data, axis, nan_policy, keepdims, **kwargs):
    # run a numpy function, using the nan version (np.nanmean etc.) for 'omit'
    arr = _check_data(data, nan_policy)
    func = getattr(np, ('nan' if nan_policy == 'omit' else '') + name)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)  # all-NaN slice -> NaN is fine
        return func(arr, axis=axis, keepdims=keepdims, **kwargs)


def _spread(name, data, axis, ddof, nan_policy, keepdims):
    # var and std: same as numpy, but NaN when there are too few values
    _check_ddof(ddof)
    result = _apply(name, data, axis, nan_policy, keepdims, ddof=ddof)
    arr = np.asarray(data, dtype=float)
    valid = ~np.isnan(arr) if nan_policy == 'omit' else np.ones(arr.shape, dtype=bool)
    count = valid.sum(axis=axis, keepdims=keepdims)
    return np.where(count > ddof, result, np.nan)


def mean(data, axis=None, *, nan_policy='propagate', keepdims=False):
    """Arithmetic mean along axis. Raises TypeError for non-numbers and
    ValueError for empty/inf data, a bad policy, or NaN with 'raise'."""
    return _apply('mean', data, axis, nan_policy, keepdims)


def median(data, axis=None, *, nan_policy='propagate', keepdims=False):
    """Median along axis (middle two are averaged for even counts)."""
    return _apply('median', data, axis, nan_policy, keepdims)


def var(data, axis=None, *, ddof=0, nan_policy='propagate', keepdims=False):
    """Variance, divided by (count - ddof). ddof=1 gives the sample variance.
    Returns NaN when count <= ddof."""
    return _spread('var', data, axis, ddof, nan_policy, keepdims)


variance = var


def std(data, axis=None, *, ddof=0, nan_policy='propagate', keepdims=False):
    """Standard deviation, same ddof rules as var()."""
    return _spread('std', data, axis, ddof, nan_policy, keepdims)


def min(data, axis=None, *, nan_policy='propagate', keepdims=False):
    """Minimum along axis."""
    return _apply('min', data, axis, nan_policy, keepdims)


def max(data, axis=None, *, nan_policy='propagate', keepdims=False):
    """Maximum along axis."""
    return _apply('max', data, axis, nan_policy, keepdims)


def quantile(data, q, axis=None, *, interpolation='linear', nan_policy='omit',
             keepdims=False):
    """Quantiles of data. q is a fraction between 0 and 1 (not 0-100),
    either a single number or a 1-D array.

    interpolation: 'linear', 'lower', 'higher' or 'midpoint'.
    The result has shape q.shape + the reduced shape of data.
    """
    q = np.asarray(q, dtype=float)
    if q.ndim > 1 or q.size == 0 or not np.all((q >= 0) & (q <= 1)):
        raise ValueError('q must be a nonempty scalar or 1-D array in [0, 1]')
    if interpolation not in ('linear', 'lower', 'higher', 'midpoint'):
        raise ValueError('unsupported interpolation method')
    return _apply('quantile', data, axis, nan_policy, keepdims,
                  q=q, method=interpolation)


def histogram(data, bins=10, range=None, *, nan_policy='omit', density=False):
    """Count values in bins. Data is flattened, NaN is skipped by default.

    bins: a positive int, or a 1-D array of strictly increasing edges.
    range: (low, high), only used when bins is an int.
    density=True scales the counts so the histogram integrates to 1.
    The last bin includes its right edge. Returns (counts, edges).
    """
    if nan_policy not in ('omit', 'raise'):
        raise ValueError("histogram nan_policy must be 'omit' or 'raise'")
    arr = _check_data(data, nan_policy).ravel()
    arr = arr[~np.isnan(arr)]

    if np.ndim(bins) == 0:
        # number of bins
        if isinstance(bins, (bool, np.bool_)) or not isinstance(bins, (int, np.integer)) or bins < 1:
            raise ValueError('bins must be a positive integer or increasing edges')
        if arr.size == 0 and range is None:
            raise ValueError('all-NaN data needs explicit edges or a range')
    else:
        # explicit edges
        bins = np.asarray(bins, dtype=float)
        if bins.ndim != 1 or bins.size < 2 or not np.isfinite(bins).all() \
                or not np.all(np.diff(bins) > 0):
            raise ValueError('bin edges must be finite and strictly increasing')

    if range is not None:
        r = np.asarray(range, dtype=float)
        if r.shape != (2,) or not np.isfinite(r).all() or r[0] >= r[1]:
            raise ValueError('range must be a finite (low, high) with low < high')

    counts, edges = np.histogram(arr, bins=bins, range=range)
    if density:
        if counts.sum() == 0:
            raise ValueError('density is undefined when no values fall in the bins')
        counts = counts / counts.sum() / np.diff(edges)
    return counts, edges


class Welford:
    """Running mean and variance without keeping the data (Welford's method).

    nan_policy: 'omit' (default) skips NaN, 'raise' throws ValueError.
    update() takes a single number or a 1-D array. For several columns use
    one Welford object per column.
    """

    def __init__(self, nan_policy='omit'):
        if nan_policy not in ('omit', 'raise'):
            raise ValueError("nan_policy must be 'omit' or 'raise'")
        self.nan_policy = nan_policy
        self.count = 0
        self._mean = 0.0
        self._m2 = 0.0  # sum of squared differences from the mean

    def update(self, values):
        """Add a number or a 1-D batch. Returns self so calls can be chained.
        If the input is rejected nothing is changed."""
        arr = np.asarray(values)
        if arr.dtype.kind not in 'biuf':
            raise TypeError('values must contain real numbers')
        if arr.ndim > 1:
            raise ValueError('values must be a scalar or 1-D')
        if arr.size == 0:
            return self
        arr = _check_data(arr, self.nan_policy).ravel()
        arr = arr[~np.isnan(arr)]

        # merge the batch into the running totals
        n = arr.size
        if n == 0:
            return self
        total = self.count + n
        delta = arr.mean() - self._mean
        self._m2 += np.sum((arr - arr.mean()) ** 2) + delta ** 2 * self.count * n / total
        self._mean += delta * n / total
        self.count = total
        return self

    @property
    def mean(self):
        """Current mean, NaN if nothing was added yet."""
        return float(self._mean) if self.count else float('nan')

    def variance(self, ddof=0):
        """Variance, NaN when count <= ddof."""
        _check_ddof(ddof)
        if self.count <= ddof:
            return float('nan')
        return float(self._m2 / (self.count - ddof))

    def std(self, ddof=0):
        """Standard deviation, NaN when count <= ddof."""
        return float(np.sqrt(self.variance(ddof)))


StreamingStats = Welford

class StreamStatistics:
    """Track per-column statistics for chunks shaped (rows, features)"""

    def __init__(self, bin_edges):
        edges = np.asarray(bin_edges, dtype=float)
        if edges.ndim != 1 or edges.size < 2:
            raise ValueError("Provide at least two histogram edges")
        if not np.isfinite(edges).all() or not np.all(edges[1:] > edges[:-1]):
            raise ValueError("Histogram edges must be finite and strictly increasing")

        self._edges = edges.copy()
        self._running = []
        self._history = []
        self._counts = None

    def update_stats(self, X_chunk):
        """Add a numeric chunk and return self."""
        X = np.asarray(X_chunk)
        if X.ndim != 2 or X.shape[0] == 0 or X.shape[1] == 0:
            raise ValueError("Expected a non-empty 2-D chunk shaped (rows, features)")
        if X.dtype.kind not in "biuf":
            raise TypeError("The chunk must contain numeric values")
        if np.isinf(X).any():
            raise ValueError("The chunk may contain NaN, but cannot contain infinity")

        n_features = X.shape[1]
        if self._counts is not None and n_features != len(self._running):
            raise ValueError("Every chunk must have the same number of feature columns")

        if self._counts is None:
            self._running = [Welford() for _ in range(n_features)]
            self._history = [[] for _ in range(n_features)]
            self._counts = np.zeros(
                (n_features, self._edges.size - 1), dtype=np.int64
            )

        for column in range(n_features):
            values = X[:, column]
            observed = values[~np.isnan(values)]
            if observed.size == 0:
                continue

            self._running[column].update(observed)
            counts, _ = histogram(observed, bins=self._edges)
            self._counts[column] += counts
            self._history[column].append(observed)

        return self

    update = update_stats

    def _check_ready(self):
        if self._counts is None:
            raise ValueError("Call update_stats(X_chunk) first")

    def mean(self):
        """Return one running mean per feature."""
        self._check_ready()
        return np.array([running.mean for running in self._running])

    def variance(self, ddof=0):
        """Return one variance per feature. Use ddof=1 for sample variance."""
        self._check_ready()
        return np.array([running.variance(ddof) for running in self._running])

    def histogram(self):
        """Return counts shaped (features, bins) and the shared bin edges."""
        self._check_ready()
        return self._counts.copy(), self._edges.copy()

    def quantile(self, q):
        """Return one exact quantile per feature. Use q=0.5 for the median."""
        self._check_ready()
        q = np.asarray(q)
        if q.ndim != 0 or q.dtype.kind not in "iuf" or not 0 <= q.item() <= 1:
            raise ValueError("q must be one number between 0 and 1")

        result = np.full(len(self._history), np.nan)
        for column, chunks in enumerate(self._history):
            if chunks:
                result[column] = quantile(np.concatenate(chunks), q.item())
        return result