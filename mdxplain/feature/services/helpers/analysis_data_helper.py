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

from ....data_selector.managers.data_selector_manager import DataSelectorManager
from ....feature_selection.managers.feature_selector_manager import FeatureSelectorManager
        

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
