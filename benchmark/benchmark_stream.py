"""Section 3: reproducible loop vs NumPy statistics/metrics benchmarks.
"""
import argparse
from pathlib import Path
import platform
import sys
import time
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from numcompute import stats, metrics


def mean_loop(values):
    total = 0.0
    for value in values:
        total += value
    return total / len(values)


def variance_loop(values):
    average = mean_loop(values)
    total = 0.0
    for value in values:
        total += (value - average) ** 2
    return total / len(values)


def histogram_loop(values, edges):
    counts = np.zeros(len(edges) - 1, dtype=int)
    for value in values:
        for i in range(len(counts)):
            if edges[i] <= value < edges[i + 1] or (i == len(counts)-1 and value == edges[-1]):
                counts[i] += 1
                break
    return counts


def accuracy_loop(actual, predicted):
    correct = 0
    for a, p in zip(actual, predicted):
        correct += int(a == p)
    return correct / len(actual)


def mse_loop(actual, predicted):
    total = 0.0
    for a, p in zip(actual, predicted):
        total += (a - p) ** 2
    return total / len(actual)


def confusion_loop(actual, predicted):
    counts = np.zeros((2, 2), dtype=int)
    for a, p in zip(actual, predicted):
        counts[a, p] += 1
    return counts


def best_time(function, repeats):
    """Warm up once, then return the smallest elapsed time in seconds."""
    function()
    timings = []
    for _ in range(repeats):
        start = time.perf_counter()
        function()
        timings.append(time.perf_counter() - start)
    return min(timings)


def benchmark_confusion_matrix():
    """Compare the same confusion-matrix calculation on the same chunks."""
    from numcompute_stream.metrics import confusion_matrix

    rng = np.random.default_rng(42)
    actual = rng.integers(0, 2, size=40_000)
    predicted = rng.integers(0, 2, size=40_000)
    chunk_size = 1_000
    chunks = [
        (actual[i:i + chunk_size], predicted[i:i + chunk_size])
        for i in range(0, actual.size, chunk_size)
    ]

    def run_loop():
        total = np.zeros((2, 2), dtype=int)
        for true_chunk, pred_chunk in chunks:
            total += confusion_loop(true_chunk, pred_chunk)
        return total

    def run_numpy():
        total = np.zeros((2, 2), dtype=int)
        for true_chunk, pred_chunk in chunks:
            total += confusion_matrix(true_chunk, pred_chunk, labels=[0, 1])
        return total

    np.testing.assert_array_equal(run_loop(), run_numpy())
    loop_seconds = best_time(run_loop, repeats=5)
    numpy_seconds = best_time(run_numpy, repeats=5)

    print("\n1. Streaming confusion matrix: Python loop vs NumPy")
    print(f"Rows: {actual.size:,} | Chunk size: {chunk_size:,} | Repeats: 5")
    print("Correctness check: both versions returned the same counts")
    print(f"Loop time:  {loop_seconds * 1000:.3f} ms")
    print(f"NumPy time: {numpy_seconds * 1000:.3f} ms")
    print(f"Speedup:    {loop_seconds / numpy_seconds:.2f}x")


def run_tree_stream(X, y, chunk_size, buffer_size, score_from):
    """Score each eligible chunk before training on its labels."""
    from numcompute_stream.tree import DecisionTreeClassifier

    tree = DecisionTreeClassifier(max_depth=3, buffer_size=buffer_size)
    correct = 0
    scored = 0
    retrain_seconds = 0.0

    for start in range(0, len(y), chunk_size):
        features = X[start:start + chunk_size]
        labels = y[start:start + chunk_size]

        if start >= score_from:
            predictions = tree.predict(features)
            correct += np.count_nonzero(predictions == labels)
            scored += labels.size

        previous_retrains = tree.n_retrains_
        before = time.perf_counter()
        tree.partial_fit(features, labels)
        elapsed = time.perf_counter() - before

        if tree.n_retrains_ > previous_retrains:
            retrain_seconds += elapsed

    return correct / scored, scored, tree.n_retrains_, retrain_seconds


def benchmark_buffer_sizes():
    """Compare three trees on identical rows in identical order."""
    rng = np.random.default_rng(42)
    X = rng.integers(0, 2, size=(2_000, 4))
    y = X[:, 0].copy()

    # Halfway through the stream, a different feature determines the label.
    # This gives the trees a change to adapt to.
    halfway = len(y) // 2
    y[halfway:] = X[halfway:, 1]

    chunk_size = 20
    buffer_sizes = (40, 100, 200)
    score_from = max(buffer_sizes)

    print("\n2. Decision tree: different buffer sizes")
    print(f"Rows: {len(y):,} | Chunk size: {chunk_size} | Max depth: 3")
    print(f"All trees are scored on the same final {len(y) - score_from:,} rows.")
    print("Time columns measure partial_fit calls that triggered a retrain.")
    print(f"{'Buffer':>7} {'Scored':>7} {'Accuracy':>10} {'Retrains':>10} "
          f"{'Total ms':>11} {'ms/retrain':>12}")

    for size in buffer_sizes:
        runs = [
            run_tree_stream(X, y, chunk_size, size, score_from)
            for _ in range(3)
        ]
        accuracy, scored, retrains, _ = runs[0]
        total_ms = 1000 * float(np.median([run[3] for run in runs]))
        print(f"{size:>7} {scored:>7} {accuracy:>10.2%} {retrains:>10} "
              f"{total_ms:>11.3f} {total_ms / retrains:>12.3f}")


if __name__ == "__main__":
    print("Assignment 2.2 streaming benchmarks")
    print(f"Python {platform.python_version()} | NumPy {np.__version__}")
    print(platform.platform())
    benchmark_confusion_matrix()
    benchmark_buffer_sizes()
