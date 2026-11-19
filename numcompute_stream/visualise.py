"""Reusable matplotlib plots for streaming metrics and predictions."""

import matplotlib.pyplot as plt
import numpy as np


def _metric_series(values):
    # Warm-up chunks may have None as their score. Plot them as gaps.
    series = np.asarray(values, dtype=float)
    if series.ndim != 1 or series.size == 0:
        raise ValueError("Provide a non-empty one-dimensional metric series")
    return series


def _finish(fig, ax, save_path, show):
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, bbox_inches="tight")
    if show:
        plt.show()
    return fig, ax


def plot_metric_over_time(metric_values, title, ylabel, *, save_path=None, show=False):
    """Plot one metric across chunks numbered from 1."""
    values = _metric_series(metric_values)
    chunks = np.arange(1, values.size + 1)

    fig, ax = plt.subplots()
    ax.plot(chunks, values, marker="o")
    ax.set(xlabel="Chunk", ylabel=ylabel, title=title)
    return _finish(fig, ax, save_path, show)


def compare_models(metric1, metric2, labels, *, save_path=None, show=False):
    """Compare two metric series measured on the same chunks."""
    first = _metric_series(metric1)
    second = _metric_series(metric2)
    if first.shape != second.shape:
        raise ValueError("Both models need one entry per chunk")
    if isinstance(labels, str) or len(labels) != 2:
        raise ValueError("Provide two model names")

    chunks = np.arange(1, first.size + 1)
    fig, ax = plt.subplots()
    ax.plot(chunks, first, marker="o", label=labels[0])
    ax.plot(chunks, second, marker="s", linestyle="--", label=labels[1])
    ax.set(xlabel="Chunk", ylabel="Metric value", title="Streaming model comparison")
    ax.legend()
    return _finish(fig, ax, save_path, show)


def plot_predictions_vs_ground_truth(y_true, y_pred, *, save_path=None, show=False):
    """Compare the predicted and actual class for each row of a chunk."""
    actual = np.asarray(y_true)
    predicted = np.asarray(y_pred)
    if actual.ndim != 1 or actual.size == 0 or actual.shape != predicted.shape:
        raise ValueError("Provide equally sized, non-empty 1-D label arrays")

    classes, positions = np.unique(
        np.concatenate((actual, predicted)), return_inverse=True
    )
    positions = positions.reshape(-1)
    rows = np.arange(1, actual.size + 1)

    fig, ax = plt.subplots()
    ax.scatter(rows - 0.1, positions[:actual.size], marker="o", label="Actual")
    ax.scatter(rows + 0.1, positions[actual.size:], marker="x", label="Predicted")
    ax.set_yticks(np.arange(classes.size))
    ax.set_yticklabels([str(label) for label in classes])
    ax.set(xlabel="Row in chunk", ylabel="Class", title="Predictions vs actual labels")
    ax.legend()
    return _finish(fig, ax, save_path, show)