import numpy as np
import pytest
from numcompute_stream.tree import DecisionTreeClassifier
def test_tree_learns_the_correct_binary_feature():
    X = [[1, 0], [1, 1], [0, 0], [0, 1]]
    y = ["no", "yes", "no", "yes"]
    tree = DecisionTreeClassifier(max_depth=1).fit(X, y)
    assert tree.predict([[0, 1], [1, 0]]).tolist() == ["yes", "no"]
def test_tree_learns_a_numeric_split():
    tree = DecisionTreeClassifier(max_depth=1)
    tree.fit([[10], [20], [30], [40]], ["low", "low", "high", "high"])
    assert tree.predict([[15], [35]]).tolist() == ["low", "high"]
def test_depth_zero_breaks_a_tie_consistently():
    tree = DecisionTreeClassifier(max_depth=0)
    tree.fit([[0], [1]], ["pass", "fail"])
    assert tree.predict([[0], [1]]).tolist() == ["fail", "fail"]
def test_small_node_predicts_the_majority_class():
    tree = DecisionTreeClassifier(min_samples_split=4)
    tree.fit([[0], [1], [1]], [0, 1, 1])
    assert tree.predict([[0], [1]]).tolist() == [1, 1]
def test_tree_waits_until_the_first_buffer_is_full():
    tree = DecisionTreeClassifier(buffer_size=3)
    tree.partial_fit([[0], [1]], [0, 1])
    assert tree.buffered_samples_ == 2
    assert tree.n_retrains_ == 0
    with pytest.raises(ValueError, match="before predicting"):
        tree.predict([[0]])
def test_exactly_filling_the_buffer_trains_the_tree():
    tree = DecisionTreeClassifier(buffer_size=3)
    tree.partial_fit([[0], [1]], [0, 1])
    tree.partial_fit([[1]], [1])
    assert tree.n_retrains_ == 1
    assert tree.buffered_samples_ == 0
    assert tree.predict([[0], [1]]).tolist() == [0, 1]
def test_crossing_the_threshold_uses_the_whole_chunk():
    tree = DecisionTreeClassifier(max_depth=0, buffer_size=4)
    tree.partial_fit([[0], [0]], [0, 0])
    tree.partial_fit([[0], [0], [0]], [1, 1, 1])
    # The fifth row gives class 1 the majority.
    assert tree.predict([[0]]).tolist() == [1]
    assert tree.buffered_samples_ == 0
def test_predictions_change_only_when_the_tree_retrains():
    tree = DecisionTreeClassifier(buffer_size=2)
    tree.partial_fit([[0], [1]], [0, 1])
    tree.partial_fit([[0]], [1])
    assert tree.predict([[0], [1]]).tolist() == [0, 1]
    tree.partial_fit([[1]], [0])
    assert tree.n_retrains_ == 2
    assert tree.predict([[0], [1]]).tolist() == [1, 0]
def test_wrong_column_count_preserves_rows_already_in_buffer():
    tree = DecisionTreeClassifier(buffer_size=2)
    tree.partial_fit([[0]], [0])
    with pytest.raises(ValueError, match="feature columns"):
        tree.partial_fit([[1, 2]], [1])
    assert tree.buffered_samples_ == 1
    tree.partial_fit([[1]], [1])
    assert tree.predict([[0], [1]]).tolist() == [0, 1]
def test_missing_feature_follows_the_no_branch():
    tree = DecisionTreeClassifier(max_depth=1)
    tree.fit([[0], [1], [0], [1]], ["no", "yes", "no", "yes"])
    assert tree.predict([[np.nan]]).tolist() == ["no"]