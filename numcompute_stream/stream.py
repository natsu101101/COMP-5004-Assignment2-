"""Manage streaming training and record one log entry per incoming chunk."""

from time import perf_counter
import tracemalloc

import numpy as np


class StreamTrainer:
    def __init__(self, pipeline):
        self.pipeline = pipeline
        self.logs = []
        self.total_correct_ = 0
        self.total_scored_ = 0

    def score_chunk(self, X, y):
        """Score the active pipeline without training it or changing the logs."""
        actual = np.asarray(y)
        predicted = self.pipeline.predict(X)
        if actual.shape != predicted.shape or actual.size == 0:
            raise ValueError("Provide one true label per predicted row")

        correct = int(np.count_nonzero(predicted == actual))
        return {
            "predictions": predicted,
            "correct": correct,
            "n_scored": actual.size,
            "accuracy": correct / actual.size,
        }

    def fit_chunk(self, X, y):
        """Predict if ready, train on the chunk, and log the result."""
        score = None
        if self.pipeline.is_fitted_:
            score = self.score_chunk(X, y)

        previous_retrains = self.pipeline.n_retrains_
        started_tracing = not tracemalloc.is_tracing()
        if started_tracing:
            tracemalloc.start()

        try:
            start = perf_counter()
            self.pipeline.partial_fit(X, y)
            elapsed = perf_counter() - start
            current_bytes, _ = tracemalloc.get_traced_memory()
        finally:
            if started_tracing:
                tracemalloc.stop()

        if score is not None:
            self.total_correct_ += score["correct"]
            self.total_scored_ += score["n_scored"]

        entry = {
            "chunk": len(self.logs) + 1,
            "rows": len(y),
            "accuracy": None if score is None else score["accuracy"],
            "cumulative_accuracy": (
                self.total_correct_ / self.total_scored_
                if self.total_scored_ else None
            ),
            "n_scored": 0 if score is None else score["n_scored"],
            "buffered_samples": self.pipeline.buffered_samples_,
            "retrained": self.pipeline.n_retrains_ > previous_retrains,
            "update_seconds": elapsed,
            "traced_memory_bytes": current_bytes,
        }
        self.logs.append(entry)
        return entry