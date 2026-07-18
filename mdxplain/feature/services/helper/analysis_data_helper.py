# mdxplain - A Python toolkit for molecular dynamics trajectory analysis
#
# Author: Maximilian Salomon
# Created with assistance from Claude Code (Claude Sonnet 4.0) and GitHub Copilot (Claude Sonnet 4.0).
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

"""Helper for analysis factory data selection operations."""

from __future__ import annotations
import time
from typing import Optional, Union, List, TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from ....pipeline.entities.pipeline_data import PipelineData

from ....data_selector.manager.data_selector_manager import DataSelectorManager
from ....feature_selection.manager.feature_selector_manager import FeatureSelectorManager
        

class AnalysisDataHelper:
    """
    Helper class for data selection in analysis factories.
    
    Provides centralized data selection logic that can be used by all
    analysis factories without code duplication. Handles trajectory
    selection and feature selection using the existing pipeline infrastructure.
    """
    
    @staticmethod
    def get_selected_data(
        pipeline_data: PipelineData,
        feature_type: str,
        feature_selector: Optional[str] = None,
        traj_selection: Optional[Union[str, int, List]] = None
    ) -> np.ndarray:
        """
        Get selected feature data using pipeline infrastructure.
        
        This method integrates trajectory selection and feature selection
        to provide the appropriate data matrix for analysis methods.

        This method creates temporary selectors as needed and cleans
        them up afterwards. If no feature_selector is provided, a temporary
        selector is created for the current feature type.

        Parameters
        ----------
        pipeline_data : PipelineData
            Pipeline data container
        feature_type : str
            Feature type name (e.g., "distances", "contacts")
        feature_selector : str, optional
            Name of existing feature selector. If None, creates temporary selector
            with all features of the current feature type
        traj_selection : str, int, list, optional
            Trajectory selection criteria. If None, uses "all"
            
        Returns
        -------
        np.ndarray
            Selected feature data matrix
            
        Raises
        ------
        ValueError
            If feature_selector contains features from other types
            or if no features of the required type are found
            
        Examples
        --------
        >>> data = AnalysisDataHelper.get_selected_data(
        ...     pipeline_data, "distances", 
        ...     feature_selector="my_selector",
        ...     traj_selection=[0, 1, 2]
        ... )
        >>> print(data.shape)  # (n_frames, n_features)
        """
        # Validate that features exist
        if feature_type not in pipeline_data.feature_data:
            raise ValueError(
                f"{feature_type.title()} features not found in pipeline. "
                f"Run pipeline.feature.add.{feature_type}() first."
            )
        
        data_selector_manager = DataSelectorManager()
        feature_selector_manager = FeatureSelectorManager()
        
        # Generate unique temporary selector names
        timestamp = str(int(time.time() * 1000000))  # microsecond timestamp
        temp_data_selector = f"_temp_data_{timestamp}_{feature_type}"
        temp_feature_selector = feature_selector if feature_selector else f"_temp_feature_{timestamp}_{feature_type}"
        
        # Create temporary data selector if trajectory selection is needed
        data_selector_to_use = None
        if traj_selection is not None and traj_selection != "all":
            # Get trajectory indices
            trajectory_indices = pipeline_data.trajectory_data.get_trajectory_indices(
                traj_selection
            )
            
            # Create frame indices dict for select_by_indices
            frame_indices = {}
            for traj_idx in trajectory_indices:
                if traj_idx in pipeline_data.feature_data[feature_type]:
                    frame_indices[traj_idx] = "all"
                else:
                    raise ValueError(
                        f"No {feature_type} features found for trajectory index {traj_idx}."
                    )
            
            # Create and configure data selector
            data_selector_manager.create(pipeline_data, temp_data_selector)
            data_selector_manager.select_by_indices(
                pipeline_data, temp_data_selector, frame_indices, mode="add"
            )
            data_selector_to_use = temp_data_selector
        
        # Create temporary feature selector if needed
        if not feature_selector:
            feature_selector_manager.create(pipeline_data, temp_feature_selector)
            feature_selector_manager.add_selection(
                pipeline_data, temp_feature_selector, feature_type, "all", common_denominator=False
            )
            
            print(
                "We are using a temporary feature selector with common_denominator=False. \n" \
                "If you give a trajectory-selection where the features have a different feature-number, \n" \
                "please create an own feature-selector and use it as parameter. \n" \
                "Otherwise we would assume a feature-combination you maybe do not want."
            )
            
            feature_selector_manager.select(pipeline_data, temp_feature_selector)
        
        # Get selected data
        data = pipeline_data.get_selected_data(
            feature_selector=temp_feature_selector,
            data_selector=data_selector_to_use
        )
        
        # Cleanup temporary selectors
        if data_selector_to_use:
            data_selector_manager.remove_selector(pipeline_data, temp_data_selector)
        if not feature_selector:
            feature_selector_manager.remove_selector(pipeline_data, temp_feature_selector)

        return data

    @staticmethod
    def get_residue_pairs(
        pipeline_data: PipelineData,
        feature_type: str,
        feature_selector: Optional[str] = None,
    ) -> tuple:
        """
        Return the residue pair of each column plus the residue count.

        The pairs are read from the reference trajectory of the selection that
        builds the pooled matrix, so per-residue reductions use the real partners
        of each residue — the ones neighbor exclusion actually kept — instead of
        assuming a full residue-residue triangle. The order matches the columns
        ``get_selected_data`` returns for the same ``feature_selector``.

        Parameters
        ----------
        pipeline_data : PipelineData
            Pipeline data container
        feature_type : str
            Feature type name (e.g. "distances", "contacts")
        feature_selector : str, optional
            Name of the feature selector whose columns to describe. None means the
            full set of the feature type, matching the temporary selector
            ``get_selected_data`` builds for the same request.

        Returns
        -------
        tuple
            (pairs, n_residues) in column order. Both are None when the pairs
            cannot be read, letting the reduction fall back to a full triangle.

        Examples
        --------
        >>> pairs, n_residues = AnalysisDataHelper.get_residue_pairs(
        ...     pipeline_data, "distances"
        ... )
        """
        reference_traj = AnalysisDataHelper._reference_trajectory(
            pipeline_data, feature_type, feature_selector
        )
        if feature_selector is not None:
            pairs = AnalysisDataHelper._selected_column_pairs(
                pipeline_data, feature_selector
            )
        else:
            pairs = AnalysisDataHelper._reference_stored_pairs(
                pipeline_data, feature_type, reference_traj
            )
        if pairs is None:
            return None, None
        return pairs, AnalysisDataHelper._residue_count(
            pipeline_data, reference_traj, pairs
        )

    @staticmethod
    def _reference_trajectory(
        pipeline_data: PipelineData,
        feature_type: str,
        feature_selector: Optional[str],
    ) -> Optional[int]:
        """
        Return the trajectory whose column layout defines the pooled matrix.

        A named selector stores the reference trajectory chosen when it was
        selected. An unfiltered request matches the temporary selector
        ``get_selected_data`` builds, whose reference defaults to the
        lowest-indexed trajectory that carries the feature.

        Parameters
        ----------
        pipeline_data : PipelineData
            Pipeline data container
        feature_type : str
            Feature type name
        feature_selector : str, optional
            Name of the feature selector, or None for the unfiltered request

        Returns
        -------
        int, optional
            Reference trajectory index, or None if a named selector has none

        Examples
        --------
        >>> reference = AnalysisDataHelper._reference_trajectory(
        ...     pipeline_data, "distances", None
        ... )
        """
        if feature_selector is not None:
            selector_data = pipeline_data.selected_feature_data[feature_selector]
            return selector_data.get_reference_trajectory()
        return min(pipeline_data.feature_data[feature_type])

    @staticmethod
    def _reference_stored_pairs(
        pipeline_data: PipelineData,
        feature_type: str,
        reference_traj: Optional[int],
    ) -> Optional[list]:
        """
        Return the positional residue pairs stored for the reference trajectory.

        The reference trajectory defines the column layout of an unfiltered
        selection, so its stored pairs describe every column in order. Returns
        None for metadata written before pairs were stored explicitly, leaving
        the full-triangle fallback to the reduction itself.

        Parameters
        ----------
        pipeline_data : PipelineData
            Pipeline data container
        feature_type : str
            Feature type name
        reference_traj : int, optional
            Reference trajectory index

        Returns
        -------
        list, optional
            Residue index pairs in column order, or None if not stored

        Examples
        --------
        >>> pairs = AnalysisDataHelper._reference_stored_pairs(
        ...     pipeline_data, "distances", 0
        ... )
        """
        if reference_traj is None:
            return None
        metadata = pipeline_data.feature_data[feature_type][
            reference_traj
        ].feature_metadata
        if metadata and metadata.get("pairs"):
            return [tuple(pair) for pair in metadata["pairs"]]
        return None

    @staticmethod
    def _selected_column_pairs(
        pipeline_data: PipelineData, feature_selector: str
    ) -> list:
        """
        Return the residue pair of each column of a feature selection.

        A selection subsets and reorders columns, so the pairs come from the
        selection's per-column metadata, read from each column's two partners.

        Parameters
        ----------
        pipeline_data : PipelineData
            Pipeline data container
        feature_selector : str
            Name of the feature selector

        Returns
        -------
        list
            Residue index pairs in selected-column order
        """
        column_meta = pipeline_data.get_selected_metadata(feature_selector)
        return [
            (
                entry["features"][0]["residue"]["index"],
                entry["features"][1]["residue"]["index"],
            )
            for entry in column_meta
        ]

    @staticmethod
    def _residue_count(
        pipeline_data: PipelineData,
        reference_traj: Optional[int],
        pairs: list,
    ) -> int:
        """
        Return the residue count that indexes the per-residue output array.

        The pairs carry positional residue indices, so the highest one plus one
        is the smallest length that indexes every residue without an out-of-range
        access. The reference trajectory's residue labels extend that length to
        the full residue count, so residues without a retained partner still get
        a slot; the pair bound stays the floor in case a label is missing.

        Parameters
        ----------
        pipeline_data : PipelineData
            Pipeline data container
        reference_traj : int, optional
            Reference trajectory index whose residue labels set the full length
        pairs : list
            Residue index pairs; the highest index sets the minimum length

        Returns
        -------
        int
            Number of residues, at least one past the highest pair index

        Examples
        --------
        >>> n_residues = AnalysisDataHelper._residue_count(
        ...     pipeline_data, 0, [(0, 1), (0, 2)]
        ... )
        """
        pair_bound = max((max(pair) for pair in pairs), default=-1) + 1
        res_label_data = pipeline_data.trajectory_data.res_label_data
        if res_label_data and reference_traj in res_label_data:
            return max(len(res_label_data[reference_traj]), pair_bound)
        return pair_bound
