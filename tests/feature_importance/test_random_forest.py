# mdxplain - A Python toolkit for molecular dynamics trajectory analysis
#
# Author: Maximilian Salomon
# Created with assistance from Claude Code (Claude Opus 4.8).
#
# Copyright (C) 2025 Maximilian Salomon
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""Unit tests for the Random Forest feature importance analyzer."""

from contextlib import nullcontext
from unittest.mock import patch

import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier

from mdxplain.feature_importance.analyzer_type import RandomForest
from mdxplain.feature_importance.analyzer_type.random_forest.random_forest_calculator import (  # noqa: E501
    RandomForestCalculator,
)


@pytest.fixture(name="binary_data")
def fixture_binary_data():
    """Return a separable binary classification dataset."""
    rng = np.random.RandomState(0)
    features = rng.rand(300, 6).astype(np.float32)
    labels = ((features[:, 1] + features[:, 4]) > 1.0).astype(int)
    return features, labels


def _make_analyzer(**kwargs) -> RandomForest:
    """Create an initialized RandomForest analyzer for tests."""
    analyzer = RandomForest(n_estimators=25, random_state=0, **kwargs)
    analyzer.init_calculator(chunk_size=64)
    return analyzer


def test_get_type_name_is_random_forest():
    """The analyzer type name identifies the Random Forest."""
    assert RandomForest.get_type_name() == "random_forest"


def test_gini_importance_shape_and_std(binary_data):
    """GINI importance returns per-feature scores and a per-tree std."""
    features, labels = binary_data
    result = _make_analyzer(importance_method="gini").compute(features, labels)

    assert result["importances"].shape == (features.shape[1],)
    assert isinstance(result["model"], RandomForestClassifier)
    assert result["metadata"]["algorithm"] == "random_forest"
    assert result["metadata"]["importance_method"] == "gini"
    assert len(result["metadata"]["importance_std"]) == features.shape[1]


def test_gini_importance_identifies_signal_features(binary_data):
    """The top-2 GINI importances are exactly the two informative features.

    The fixture builds ``y = (feature_1 + feature_4) > 1.0``, so only features
    1 and 4 carry signal while features 0, 2, 3 and 5 are noise. The two
    highest importances must therefore be features 1 and 4.
    """
    features, labels = binary_data
    result = _make_analyzer(importance_method="gini").compute(features, labels)
    top_two = np.argsort(result["importances"])[-2:]
    assert {int(index) for index in top_two} == {1, 4}


def test_shap_importance_shape_and_nonneg(binary_data):
    """SHAP importance returns non-negative per-feature scores."""
    features, labels = binary_data
    result = _make_analyzer(importance_method="shap").compute(features, labels)

    assert result["importances"].shape == (features.shape[1],)
    assert np.all(result["importances"] >= 0)
    assert result["metadata"]["importance_method"] == "shap"


def test_shap_multiclass_shape():
    """SHAP importance handles multi-class targets."""
    rng = np.random.RandomState(1)
    features = rng.rand(200, 5).astype(np.float32)
    labels = rng.randint(0, 3, size=200)
    result = _make_analyzer(importance_method="shap").compute(features, labels)

    assert result["importances"].shape == (features.shape[1],)
    assert result["metadata"]["n_classes"] == 3


def test_prepare_shap_eval_set_caps_rows(binary_data):
    """_prepare_shap_eval_set limits SHAP evaluation to shap_sample_size rows.

    With a cap below the row count it returns exactly that many rows; with no
    cap (or a cap at least as large as the data) it returns every row.
    """
    features, labels = binary_data
    capped = RandomForestCalculator._prepare_shap_eval_set(
        features, labels, 50, 0
    )
    assert capped.shape[0] == 50

    uncapped = RandomForestCalculator._prepare_shap_eval_set(
        features, labels, None, 0
    )
    assert uncapped.shape[0] == features.shape[0]

    cap_above = RandomForestCalculator._prepare_shap_eval_set(
        features, labels, 500, 0
    )
    assert cap_above.shape[0] == features.shape[0]


def test_compute_invokes_shap_eval_set_capping(binary_data):
    """compute routes SHAP through _prepare_shap_eval_set and caps its rows.

    Spies on _prepare_shap_eval_set during a real compute() call to confirm it
    is invoked with the configured shap_sample_size and actually reduces the
    evaluation set to that many rows.
    """
    features, labels = binary_data
    analyzer = _make_analyzer(importance_method="shap", shap_sample_size=50)

    captured = {}
    original = RandomForestCalculator._prepare_shap_eval_set

    def spy(x_train, y_train, shap_sample_size, random_state):
        eval_set = original(x_train, y_train, shap_sample_size, random_state)
        captured["shap_sample_size"] = shap_sample_size
        captured["rows"] = eval_set.shape[0]
        return eval_set

    with patch.object(
        RandomForestCalculator, "_prepare_shap_eval_set", side_effect=spy
    ):
        analyzer.compute(features, labels)

    assert captured["shap_sample_size"] == 50
    assert captured["rows"] == 50


def test_invalid_input_raises(binary_data):
    """Invalid input data raises a clear ValueError."""
    features, labels = binary_data
    analyzer = _make_analyzer()

    with pytest.raises(ValueError):
        analyzer.compute(features[:, 0], labels)

    nan_features = features.copy()
    nan_features[0, 0] = np.nan
    with pytest.raises(ValueError):
        analyzer.compute(nan_features, labels)


def test_compute_invokes_stratified_sampling(binary_data):
    """compute samples the training data via _apply_stratified_sampling.

    Spies on _apply_stratified_sampling during a real compute() call to confirm
    it is invoked with the configured max_samples, actually reduces the data to
    that many rows, and that the sampled data is what training reports.
    """
    features, labels = binary_data
    analyzer = _make_analyzer(importance_method="gini", max_samples=120)

    captured = {}
    original = RandomForestCalculator._apply_stratified_sampling

    def spy(x, y, max_samples, random_state=None):
        x_sampled, y_sampled = original(x, y, max_samples, random_state)
        captured["max_samples"] = max_samples
        captured["rows"] = x_sampled.shape[0]
        return x_sampled, y_sampled

    with patch.object(
        RandomForestCalculator, "_apply_stratified_sampling", side_effect=spy
    ):
        result = analyzer.compute(features, labels)

    assert captured["max_samples"] == 120
    assert captured["rows"] == 120
    assert result["metadata"]["n_samples"] == 120
    assert result["metadata"]["sampling"]["sampled"] is True


def test_apply_stratified_sampling_caps_rows(binary_data):
    """_apply_stratified_sampling limits training to max_samples rows.

    Above the cap it returns exactly max_samples rows for both features and
    labels; at or below the cap it returns the data unchanged.
    """
    features, labels = binary_data
    x_capped, y_capped = RandomForestCalculator._apply_stratified_sampling(
        features, labels, 120, 0
    )
    assert x_capped.shape[0] == 120
    assert y_capped.shape[0] == 120

    x_full, y_full = RandomForestCalculator._apply_stratified_sampling(
        features, labels, 500, 0
    )
    assert x_full.shape[0] == features.shape[0]
    assert y_full.shape[0] == features.shape[0]


def test_compute_requires_initialized_calculator(binary_data):
    """compute raises when the calculator was not initialized."""
    features, labels = binary_data
    analyzer = RandomForest(n_estimators=10)

    with pytest.raises(ValueError):
        analyzer.compute(features, labels)


def test_get_params_returns_all_parameters():
    """get_params returns the complete reproducibility parameter set."""
    analyzer = RandomForest(
        n_estimators=42,
        importance_method="shap",
        max_depth=7,
        random_state=1,
    )
    params = analyzer.get_params()

    expected_keys = {
        "n_estimators",
        "importance_method",
        "criterion",
        "max_depth",
        "min_samples_split",
        "min_samples_leaf",
        "min_weight_fraction_leaf",
        "max_features",
        "bootstrap",
        "oob_score",
        "random_state",
        "max_leaf_nodes",
        "min_impurity_decrease",
        "class_weight",
        "ccp_alpha",
    }
    assert set(params.keys()) == expected_keys
    assert params["n_estimators"] == 42
    assert params["importance_method"] == "shap"
    assert params["max_depth"] == 7
    assert params["random_state"] == 1


def test_resolve_blas_limit_applies_thread_policy():
    """_resolve_blas_limit follows the configured BLAS thread policy."""
    auto = RandomForestCalculator(max_blas_threads=1, auto_limit_blas=True)
    assert auto._resolve_blas_limit(-1) == 1
    assert auto._resolve_blas_limit(1) == 1

    auto_four = RandomForestCalculator(max_blas_threads=4, auto_limit_blas=True)
    assert auto_four._resolve_blas_limit(1) == 4
    assert auto_four._resolve_blas_limit(-1) == 1

    disabled = RandomForestCalculator(
        max_blas_threads=None, auto_limit_blas=False
    )
    assert disabled._resolve_blas_limit(2) is None


def test_uses_parallel_jobs():
    """_uses_parallel_jobs is True only for more than one worker."""
    assert RandomForestCalculator._uses_parallel_jobs(-1) is True
    assert RandomForestCalculator._uses_parallel_jobs(2) is True
    assert RandomForestCalculator._uses_parallel_jobs(1) is False
    assert RandomForestCalculator._uses_parallel_jobs(None) is False


def test_limit_threadpools_selects_context_manager():
    """_limit_threadpools limits threads only when the policy requires it."""
    disabled = RandomForestCalculator(
        max_blas_threads=None, auto_limit_blas=False
    )
    assert isinstance(disabled._limit_threadpools(2), nullcontext)

    limiting = RandomForestCalculator(max_blas_threads=1, auto_limit_blas=True)
    assert not isinstance(limiting._limit_threadpools(-1), nullcontext)


def test_aggregate_abs_shap_reduces_class_axis():
    """SHAP aggregation averages the class axis to per-feature values."""
    shap_values = np.ones((4, 3, 2))
    reduced = RandomForestCalculator._aggregate_abs_shap(shap_values)
    assert reduced.shape == (4, 3)
    assert np.allclose(reduced, 1.0)
