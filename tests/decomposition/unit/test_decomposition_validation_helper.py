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

"""Unit tests for DecompositionValidationHelper."""

import numpy as np
import pytest

from mdxplain.decomposition.decomposition_type import PCA, DiffusionMaps
from mdxplain.decomposition.entities.decomposition_data import (
    DecompositionData,
)
from mdxplain.decomposition.helper.decomposition_validation_helper import (
    DecompositionValidationHelper,
)
from mdxplain.feature_selection.entities.feature_selector_data import (
    FeatureSelectorData,
)
from mdxplain.pipeline.entities.pipeline_data import PipelineData


def _decomposition(n_components=30):
    """Build a DecompositionData with a data matrix of the given width."""
    data = DecompositionData("kernel_pca")
    data.data = np.zeros((10, n_components))
    return data


def _pipeline_with(decompositions=None, selections=None):
    """Build a PipelineData populated with the given data mappings."""
    pipe = PipelineData()
    for name, decomposition in (decompositions or {}).items():
        pipe.decomposition_data[name] = decomposition
    for name, feature_types in (selections or {}).items():
        selection = FeatureSelectorData(name)
        for feature_type in feature_types:
            selection.selections[feature_type] = []
        pipe.selected_feature_data[name] = selection
    return pipe


def test_validate_source_exists_passes_when_present():
    """A present source name does not raise."""
    pipe = _pipeline_with(decompositions={"src": _decomposition()})
    DecompositionValidationHelper.validate_source_exists(pipe, "src")


def test_validate_source_exists_raises_when_missing():
    """A missing source name raises ValueError."""
    pipe = _pipeline_with(decompositions={"src": _decomposition()})
    with pytest.raises(ValueError, match="not found"):
        DecompositionValidationHelper.validate_source_exists(pipe, "other")


def test_validate_source_exists_lists_available_names():
    """The error message lists the exact available decompositions."""
    pipe = _pipeline_with(
        decompositions={"a": _decomposition(), "b": _decomposition()}
    )
    with pytest.raises(ValueError) as exc:
        DecompositionValidationHelper.validate_source_exists(pipe, "missing")
    message = str(exc.value)
    assert "Decomposition 'missing' not found." in message
    assert "['a', 'b']" in message


def test_validate_target_available_passes_when_free():
    """A free target name does not raise."""
    pipe = _pipeline_with(decompositions={"src": _decomposition()})
    DecompositionValidationHelper.validate_target_available(pipe, "new", False)


def test_validate_target_available_raises_when_taken():
    """A taken target name raises when force is False."""
    pipe = _pipeline_with(decompositions={"src": _decomposition()})
    with pytest.raises(ValueError) as exc:
        DecompositionValidationHelper.validate_target_available(
            pipe, "src", False
        )
    assert "Decomposition 'src' already exists." in str(exc.value)


def test_validate_target_available_force_allows_overwrite():
    """A taken target name is allowed when force is True."""
    pipe = _pipeline_with(decompositions={"src": _decomposition()})
    DecompositionValidationHelper.validate_target_available(pipe, "src", True)


def test_validate_component_count_passes_in_range():
    """A count within range does not raise."""
    DecompositionValidationHelper.validate_component_count(
        _decomposition(30), 5
    )


def test_validate_component_count_allows_boundaries():
    """The full count and a single component are both valid."""
    DecompositionValidationHelper.validate_component_count(
        _decomposition(30), 30
    )
    DecompositionValidationHelper.validate_component_count(
        _decomposition(30), 1
    )


def test_validate_component_count_raises_when_no_data():
    """A source without computed data raises ValueError."""
    empty = DecompositionData("pca")
    with pytest.raises(ValueError, match="no computed data"):
        DecompositionValidationHelper.validate_component_count(empty, 5)


def test_validate_component_count_raises_below_one():
    """A count below one raises ValueError."""
    with pytest.raises(ValueError, match="between 1"):
        DecompositionValidationHelper.validate_component_count(
            _decomposition(30), 0
        )


def test_validate_component_count_raises_above_available():
    """A count above the available components raises ValueError."""
    with pytest.raises(ValueError, match="between 1"):
        DecompositionValidationHelper.validate_component_count(
            _decomposition(30), 31
        )


def test_validate_chunk_size_passes_for_positive_int():
    """A positive integer chunk size does not raise."""
    DecompositionValidationHelper.validate_chunk_size(1000)


def test_validate_chunk_size_raises_for_zero():
    """A zero chunk size raises ValueError."""
    with pytest.raises(ValueError, match="positive integer"):
        DecompositionValidationHelper.validate_chunk_size(0)


def test_validate_chunk_size_raises_for_negative():
    """A negative chunk size raises ValueError."""
    with pytest.raises(ValueError, match="positive integer"):
        DecompositionValidationHelper.validate_chunk_size(-5)


def test_validate_decomposition_type_passes_for_real_type():
    """A real decomposition type does not raise."""
    DecompositionValidationHelper.validate_decomposition_type(PCA())


def test_validate_feature_type_compatibility_passes_when_unrestricted():
    """A type with no required feature type does not raise."""
    pipe = _pipeline_with()
    DecompositionValidationHelper.validate_feature_type_compatibility(
        pipe, "sel", PCA()
    )


def test_validate_feature_type_compatibility_raises_when_selection_missing():
    """A required type with a missing selection raises ValueError."""
    pipe = _pipeline_with()
    with pytest.raises(ValueError, match="Feature selection 'sel' not found"):
        DecompositionValidationHelper.validate_feature_type_compatibility(
            pipe, "sel", DiffusionMaps(n_components=2)
        )


def test_validate_feature_type_compatibility_passes_when_all_match():
    """A selection with only the required feature type does not raise."""
    pipe = _pipeline_with(selections={"sel": ["coordinates"]})
    DecompositionValidationHelper.validate_feature_type_compatibility(
        pipe, "sel", DiffusionMaps(n_components=2)
    )


def test_validate_feature_type_compatibility_raises_on_incompatible():
    """A selection with a foreign feature type raises a descriptive error."""
    pipe = _pipeline_with(selections={"sel": ["contacts"]})
    with pytest.raises(ValueError) as exc:
        DecompositionValidationHelper.validate_feature_type_compatibility(
            pipe, "sel", DiffusionMaps(n_components=2)
        )
    message = str(exc.value)
    assert "requires features of type 'coordinates'" in message
    assert "contacts" in message
