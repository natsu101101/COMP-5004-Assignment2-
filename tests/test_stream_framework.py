import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

from numcompute_stream.metrics import StreamingMetrics
from numcompute_stream.pipeline import Pipeline
from numcompute_stream.preprocessing import OneHotEncoder, StandardScaler
from numcompute_stream.stats import StreamStatistics
from numcompute_stream.stream import StreamTrainer
from numcompute_stream.tree import DecisionTreeClassifier
from numcompute_stream.visualise import (
    compare_models,
    plot_metric_over_time,
    plot_predictions_vs_ground_truth,
)


def make_pipeline(buffer_size=4):
    return Pipeline([
        ("scale", StandardScaler()),
        ("tree", DecisionTreeClassifier(max_depth=2, buffer_size=buffer_size)),
    ])


# Statistics

def test_running_statistics_match_the_combined_data():
    stats = StreamStatistics([0, 5, 10])
    stats.update_stats([[2, 4], [4, np.nan]])
    stats.update_stats([[6, 8]])

    np.testing.assert_allclose(stats.mean(), [4, 6])
    np.testing.assert_allclose(stats.variance(), [8 / 3, 4])
    np.testing.assert_allclose(stats.quantile(0.5), [4, 6])


def test_histogram_uses_the_same_edges_across_chunks():
    stats = StreamStatistics([0, 5, 10])
    stats.update_stats([[0], [4]])
    stats.update_stats([[5], [10], [np.nan]])

    counts, edges = stats.histogram()
    np.testing.assert_array_equal(edges, [0, 5, 10])
    np.testing.assert_array_equal(counts, [[2, 2]])


def test_all_missing_column_can_learn_from_a_later_chunk():
    stats = StreamStatistics([0, 5, 10])
    stats.update_stats([[np.nan], [np.nan]])

    assert np.isnan(stats.mean()[0])
    assert np.isnan(stats.quantile(0.5)[0])

    stats.update_stats([[6]])
    np.testing.assert_allclose(stats.mean(), [6])
    np.testing.assert_array_equal(stats.histogram()[0], [[0, 1]])


# Metrics

def test_cumulative_metrics_count_rows_across_unequal_chunks():
    metrics = StreamingMetrics(average="binary")
    metrics.update([0, 1, 1], [0, 1, 0])
    metrics.update([0], [1])
    result = metrics.result()

    assert result["n_samples"] == 4
    assert result["accuracy"] == pytest.approx(0.5)
    assert result["precision"] == pytest.approx(0.5)
    assert result["recall"] == pytest.approx(0.5)
    assert result["f1"] == pytest.approx(0.5)
    np.testing.assert_array_equal(result["confusion_matrix"], [[1, 1], [1, 1]])


def test_rolling_window_can_cut_through_an_older_chunk():
    metrics = StreamingMetrics(window_size=3)
    metrics.update([0, 0, 1], [1, 1, 1])
    metrics.update([0, 1], [0, 1])

    # The three most recent predictions are all correct.
    result = metrics.result()
    assert result["n_samples"] == 3
    assert result["accuracy"] == 1.0


def test_auc_combines_scores_from_different_chunks():
    metrics = StreamingMetrics()
    metrics.update([0, 0], [0, 0], [0.1, 0.6])
    assert np.isnan(metrics.result()["auc"])

    metrics.update([1, 1], [1, 1], [0.4, 0.9])

    # Three of the four positive-negative pairs are ranked correctly.
    assert metrics.result()["auc"] == pytest.approx(0.75)


def test_reset_clears_the_previous_metric_history():
    metrics = StreamingMetrics()
    metrics.update([0, 1], [0, 1])
    metrics.reset()

    with pytest.raises(ValueError, match="update"):
        metrics.result()

    metrics.update([1], [0])
    assert metrics.result()["n_samples"] == 1
    assert metrics.result()["accuracy"] == 0.0


# Pipeline and trainer

def test_pipeline_waits_then_trains_on_the_whole_buffer():
    pipe = make_pipeline(buffer_size=4)
    pipe.partial_fit([[0], [1]], [0, 1])

    assert pipe.buffered_samples_ == 2
    with pytest.raises(ValueError, match="buffer"):
        pipe.predict([[0]])

    # This chunk takes the total from two rows to five.
    pipe.partial_fit([[0], [1], [1]], [0, 1, 1])

    assert pipe.buffered_samples_ == 0
    assert pipe.n_retrains_ == 1
    np.testing.assert_array_equal(pipe.predict([[0], [1]]), [0, 1])
    np.testing.assert_allclose(pipe.named_steps["scale"].mean_, [0.6])


def test_pending_data_does_not_change_active_scaling_or_predictions():
    pipe = make_pipeline(buffer_size=4)
    pipe.partial_fit([[0], [0], [10], [10]], [0, 0, 1, 1])

    before = pipe.predict([[0], [10]])
    pipe.partial_fit([[1000]], [0])

    assert pipe.buffered_samples_ == 1
    np.testing.assert_allclose(pipe.named_steps["scale"].mean_, [5])
    np.testing.assert_array_equal(pipe.predict([[0], [10]]), before)


def test_encoder_and_tree_share_columns_when_a_new_category_arrives():
    pipe = Pipeline([
        ("encode", OneHotEncoder()),
        ("tree", DecisionTreeClassifier(buffer_size=4)),
    ])
    pipe.partial_fit([["red"], ["red"]], [0, 0])
    pipe.partial_fit([["blue"], ["blue"]], [1, 1])

    np.testing.assert_array_equal(pipe.predict([["blue"], ["red"]]), [1, 0])


def test_trainer_scores_before_retraining_and_weights_accuracy_by_rows():
    trainer = StreamTrainer(make_pipeline(buffer_size=2))

    first = trainer.fit_chunk([[0], [1]], [0, 1])
    assert first["retrained"]
    assert first["accuracy"] is None
    assert first["n_scored"] == 0

    # The old tree gets both reversed labels wrong before it retrains.
    second = trainer.fit_chunk([[0], [1]], [1, 0])
    assert second["accuracy"] == 0.0
    assert second["retrained"]

    third = trainer.fit_chunk([[0]], [1])
    assert third["accuracy"] == 1.0
    assert third["cumulative_accuracy"] == pytest.approx(1 / 3)
    assert third["traced_memory_bytes"] >= 0
    assert third["update_seconds"] >= 0
    assert len(trainer.logs) == 3


# Visualisation

def test_metric_plot_preserves_warmup_gap_and_saves_a_file(tmp_path):
    destination = tmp_path / "accuracy.png"
    fig, ax = plot_metric_over_time(
        [None, 0.5, 1.0], "Streaming accuracy", "Accuracy",
        save_path=destination,
    )
    try:
        np.testing.assert_array_equal(ax.lines[0].get_xdata(), [1, 2, 3])
        values = ax.lines[0].get_ydata()
        assert np.isnan(values[0])
        np.testing.assert_allclose(values[1:], [0.5, 1.0])
        assert destination.stat().st_size > 0
    finally:
        plt.close(fig)


def test_model_comparison_plots_both_named_series():
    fig, ax = compare_models([0.5, 1.0], [0.5, 0.75], ["Small", "Large"])
    try:
        assert [line.get_label() for line in ax.lines] == ["Small", "Large"]
        np.testing.assert_allclose(ax.lines[1].get_ydata(), [0.5, 0.75])
    finally:
        plt.close(fig)


def test_prediction_plot_displays_actual_and_predicted_classes():
    fig, ax = plot_predictions_vs_ground_truth(
        ["fail", "pass"], ["fail", "fail"]
    )
    try:
        assert [item.get_label() for item in ax.collections] == [
            "Actual", "Predicted"
        ]
        assert [tick.get_text() for tick in ax.get_yticklabels()] == [
            "fail", "pass"
        ]
    finally:
        plt.close(fig)