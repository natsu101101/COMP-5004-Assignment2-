import numpy as np
import pytest
from numcompute_stream.preprocessing import (
    MinMaxScaler,
    OneHotEncoder,
    SimpleImputer,
    StandardScaler,
)
def test_standard_scaler_matches_all_rows_combined():
    scaler = StandardScaler()
    scaler.partial_fit([[2, 10], [4, 20], [6, 30]])
    scaler.partial_fit([[12, 60]])
    all_rows = np.array([[2, 10], [4, 20], [6, 30], [12, 60]])
    np.testing.assert_allclose(scaler.mean_, all_rows.mean(axis=0))
    np.testing.assert_allclose(scaler.var_, all_rows.var(axis=0))
    np.testing.assert_array_equal(scaler.count_, [4, 4])
def test_standard_scaler_handles_missing_and_constant_columns():
    scaler = StandardScaler()
    scaler.partial_fit([[np.nan, 7], [np.nan, 7]])
    scaler.partial_fit([[4, 7], [8, np.nan]])
    np.testing.assert_array_equal(scaler.count_, [2, 3])
    np.testing.assert_allclose(scaler.transform([[8, 7]]), [[1, 0]])
def test_fit_starts_fresh_after_previous_chunks():
    scaler = StandardScaler()
    scaler.partial_fit([[100], [200]])
    scaler.fit([[2], [4]])
    np.testing.assert_allclose(scaler.mean_, [3])
    np.testing.assert_array_equal(scaler.count_, [2])
def test_wrong_column_count_preserves_previous_statistics():
    scaler = StandardScaler().partial_fit([[2], [4]])
    with pytest.raises(ValueError):
        scaler.partial_fit([[10, 20]])
    np.testing.assert_allclose(scaler.mean_, [3])
    np.testing.assert_array_equal(scaler.count_, [2])
def test_minmax_scaler_updates_both_ends_of_the_range():
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaler.partial_fit([[2], [4]])
    scaler.partial_fit([[0], [8]])
    np.testing.assert_allclose(scaler.data_min_, [0])
    np.testing.assert_allclose(scaler.data_max_, [8])
    np.testing.assert_allclose(scaler.transform([[0], [4], [8]]), [[-1], [0], [1]])
def test_minmax_scaler_can_learn_a_column_that_started_missing():
    scaler = MinMaxScaler()
    scaler.partial_fit([[np.nan, 5], [np.nan, 5]])
    scaler.partial_fit([[2, 5], [6, 5]])
    np.testing.assert_allclose(scaler.transform([[4, 5]]), [[0.5, 0]])
def test_mean_imputer_uses_observed_values_across_chunks():
    imputer = SimpleImputer(strategy="mean", fill_value=-1)
    imputer.partial_fit([[np.nan, 10], [np.nan, 20]])
    np.testing.assert_allclose(imputer.transform([[np.nan, np.nan]]), [[-1, 15]])
    imputer.partial_fit([[8, 30]])
    np.testing.assert_allclose(imputer.transform([[np.nan, np.nan]]), [[8, 20]])
def test_constant_imputer_only_replaces_missing_values():
    imputer = SimpleImputer(strategy="constant", fill_value=-5)
    imputer.partial_fit([[1, np.nan]])
    imputer.partial_fit([[3, 4]])
    result = imputer.transform([[np.nan, 9], [7, np.nan]])
    np.testing.assert_allclose(result, [[-5, 9], [7, -5]])
def test_new_categories_keep_old_output_columns_in_place():
    encoder = OneHotEncoder()
    encoder.partial_fit([["red", "small"]])
    before = encoder.transform([["red", "small"]])
    encoder.partial_fit([["blue", "large"]])
    after = encoder.transform([["red", "small"]])
    np.testing.assert_array_equal(after[:, :2], before)
    np.testing.assert_array_equal(after[:, 2:], [[0, 0]])
    np.testing.assert_array_equal(
        encoder.transform([["blue", "large"]]), [[0, 0, 1, 1]]
    )
def test_encoder_handles_an_unknown_category_using_its_setting():
    strict = OneHotEncoder(handle_unknown="error").fit([["red"]])
    with pytest.raises(ValueError):
        strict.transform([["blue"]])
    tolerant = OneHotEncoder(handle_unknown="ignore").fit([["red"]])
    np.testing.assert_array_equal(tolerant.transform([["blue"]]), [[0]])
    tolerant.partial_fit([["blue"]])
    np.testing.assert_array_equal(tolerant.transform([["blue"]]), [[0, 1]])