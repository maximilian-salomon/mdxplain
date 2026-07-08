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

"""
Validation helper for decomposition management.

This module provides the DecompositionValidationHelper class with static
methods for validating decomposition inputs. Extracted from
DecompositionManager to improve code organization and testability.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, List

if TYPE_CHECKING:
    from ...pipeline.entities.pipeline_data import PipelineData
    from ..decomposition_type.interfaces.decomposition_type_base import (
        DecompositionTypeBase,
    )
    from ..entities.decomposition_data import DecompositionData


class DecompositionValidationHelper:
    """
    Static helper class for decomposition validation operations.

    Provides validation methods for decomposition names, component counts,
    chunk size, decomposition type instances, and feature-type compatibility.
    All methods are static and stateless.
    """

    @staticmethod
    def validate_source_exists(
        pipeline_data: "PipelineData", source_name: str
    ) -> None:
        """
        Validate that a decomposition with the given name exists.

        Parameters
        ----------
        pipeline_data : PipelineData
            Pipeline data object containing decomposition data
        source_name : str
            Name of the decomposition to validate

        Returns
        -------
        None
            Method returns nothing, raises ValueError if the source is missing

        Raises
        ------
        ValueError
            If no decomposition with ``source_name`` exists
        """
        if source_name not in pipeline_data.decomposition_data:
            available = list(pipeline_data.decomposition_data.keys())
            raise ValueError(
                f"Decomposition '{source_name}' not found. "
                f"Available: {available}"
            )

    @staticmethod
    def validate_target_available(
        pipeline_data: "PipelineData", new_name: str, force: bool
    ) -> None:
        """
        Validate that a target name is free or overwriting is allowed.

        Parameters
        ----------
        pipeline_data : PipelineData
            Pipeline data object containing decomposition data
        new_name : str
            Target name for the new decomposition
        force : bool
            Whether overwriting an existing target is allowed

        Returns
        -------
        None
            Method returns nothing, raises ValueError if the name is taken

        Raises
        ------
        ValueError
            If ``new_name`` already exists and ``force`` is False
        """
        if new_name in pipeline_data.decomposition_data and not force:
            raise ValueError(
                f"Decomposition '{new_name}' already exists. "
                "Use force=True to overwrite."
            )

    @staticmethod
    def validate_component_count(
        source: "DecompositionData", n_components: int
    ) -> None:
        """
        Validate the requested component count against the source.

        This is important for the reducing (for example PCA reduce)
        of the number of components of the decomposition.

        Parameters
        ----------
        source : DecompositionData
            Source decomposition to reduce
        n_components : int
            Number of leading components requested

        Returns
        -------
        None
            Method returns nothing, raises ValueError if the count is invalid

        Raises
        ------
        ValueError
            If the source has no computed data or ``n_components`` is outside
            the range of available components
        """
        if source.data is None:
            raise ValueError(
                "Source decomposition has no computed data to reduce."
            )
        available = source.data.shape[1]
        if n_components < 1 or n_components > available:
            raise ValueError(
                f"n_components ({n_components}) must be between 1 and the "
                f"{available} available components (they were never computed)."
            )

    @staticmethod
    def validate_chunk_size(chunk_size: int) -> None:
        """
        Validate that the chunk size is a positive integer.

        Parameters
        ----------
        chunk_size : int
            Chunk size for memory-mapped processing

        Returns
        -------
        None
            Method returns nothing, raises ValueError if the size is invalid

        Raises
        ------
        ValueError
            If the chunk size is not a positive integer
        """
        if not isinstance(chunk_size, int) or chunk_size <= 0:
            raise ValueError("Chunk size must be a positive integer.")

    @staticmethod
    def validate_decomposition_type(
        decomposition_type: "DecompositionTypeBase",
    ) -> None:
        """
        Validate that the object is a decomposition type instance.

        Parameters
        ----------
        decomposition_type : DecompositionTypeBase
            Object expected to expose ``init_calculator``

        Returns
        -------
        None
            Method returns nothing, raises ValueError if the object is invalid

        Raises
        ------
        ValueError
            If the object does not expose ``init_calculator``
        """
        if not hasattr(decomposition_type, "init_calculator"):
            raise ValueError(
                f"Invalid decomposition type '{decomposition_type}'. "
                "Please provide a decomposition type instance."
            )

    @staticmethod
    def validate_feature_type_compatibility(
        pipeline_data: "PipelineData",
        selection_name: str,
        decomposition_type: "DecompositionTypeBase",
    ) -> None:
        """
        Validate that a decomposition type matches the selected feature type.

        Some decomposition methods require a specific feature type (for
        example DiffusionMaps requires coordinates). When the type imposes a
        requirement, every feature in the selection must match it.

        Parameters
        ----------
        pipeline_data : PipelineData
            Pipeline data object containing the feature selection
        selection_name : str
            Name of the feature selection
        decomposition_type : DecompositionTypeBase
            Decomposition type to validate

        Returns
        -------
        None
            Method returns nothing, raises ValueError on incompatibility

        Raises
        ------
        ValueError
            If the selection is missing or contains incompatible feature types
        """
        required_type = decomposition_type.get_required_feature_type()
        if required_type is None:
            return
        if selection_name not in pipeline_data.selected_feature_data:
            raise ValueError(f"Feature selection '{selection_name}' not found")
        selection_data = pipeline_data.selected_feature_data[selection_name]
        incompatible = (
            DecompositionValidationHelper._incompatible_feature_types(
                selection_data, required_type
            )
        )
        if incompatible:
            DecompositionValidationHelper._raise_incompatible_features(
                decomposition_type, selection_name, required_type, incompatible
            )

    @staticmethod
    def _incompatible_feature_types(
        selection_data: Any, required_type: str
    ) -> List[str]:
        """
        Collect feature types in a selection that differ from the requirement.

        Parameters
        ----------
        selection_data : Any
            Selection data whose ``selections`` maps feature types to entries
        required_type : str
            Feature type the decomposition requires

        Returns
        -------
        List[str]
            Feature types present in the selection that are not the required
            type
        """
        return [
            feature_type
            for feature_type in selection_data.selections
            if feature_type != required_type
        ]

    @staticmethod
    def _raise_incompatible_features(
        decomposition_type: "DecompositionTypeBase",
        selection_name: str,
        required_type: str,
        incompatible: List[str],
    ) -> None:
        """
        Raise a ValueError describing incompatible feature types.

        Parameters
        ----------
        decomposition_type : DecompositionTypeBase
            Decomposition type that imposed the requirement
        selection_name : str
            Name of the feature selection
        required_type : str
            Feature type the decomposition requires
        incompatible : List[str]
            Feature types that violate the requirement

        Returns
        -------
        None
            Method never returns, it always raises

        Raises
        ------
        ValueError
            Always, describing the incompatible feature types
        """
        decomposition_name = decomposition_type.__class__.__name__
        incompatible_list = ", ".join(incompatible)
        raise ValueError(
            f"{decomposition_name} requires features of type "
            f"'{required_type}' only. Selection '{selection_name}' contains "
            f"incompatible features: {incompatible_list}"
        )
