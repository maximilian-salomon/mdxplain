# mdxplain - A Python toolkit for molecular dynamics trajectory analysis
#
# Author: Maximilian Salomon
# Created with assistance from Claude-4-Sonnet and Cursor AI.
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
Feature selector for creating custom feature data matrices.

Provides a simple selection interface using structured metadata.
"""

import warnings
from typing import Dict, List, Union

from mdxplain.data.helper.feature_selector_helpers import FeatureSelectorHelper


class FeatureSelector:
    """
    Simple selector for creating custom feature data matrices.
    
    Uses structured metadata for efficient feature selection.
    
    Examples:
    ---------
    >>> selector = FeatureSelector()
    >>> selector.add("distances", "res ALA")
    >>> selector.add("distances", "resid 123-140")
    >>> selector.select(traj_data, "my_analysis")
    """
    
    def __init__(self):
        """Initialize feature selector with empty selections."""
        self.selections: Dict[str, List[dict]] = {}
    
    def add(self, feature_type: Union[str, object], selection: str = "all", use_reduced: bool = False):
        """
        Add a feature type with selection criteria.
        
        Parameters:
        -----------
        feature_type : str or object
            Feature type (e.g., "distances", "contacts")
        selection : str, default="all"
            Selection string
        use_reduced : bool, default=False
            Whether to use reduced data (True) or original data (False)
        """
        feature_key = self._convert_feature_type_to_key(feature_type)
        
        if feature_key not in self.selections:
            self.selections[feature_key] = []
        
        self.selections[feature_key].append({
            'selection': selection,
            'use_reduced': use_reduced
        })
    
    def select(self, traj_data, name: str):
        """
        Apply selections and store results in trajectory data.
        
        Parameters:
        -----------
        traj_data : TrajectoryData
            Trajectory data object containing computed features
        name : str
            Name to store the selection under
        """
        self._validate_features_exist(traj_data)
        self._initialize_selected_data_storage(traj_data, name)
        self._process_all_selections(traj_data, name)
    
    def _convert_feature_type_to_key(self, feature_type: Union[str, object]) -> str:
        """Convert feature_type to string key."""
        if isinstance(feature_type, str):
            return feature_type
        elif hasattr(feature_type, 'get_type_name'):
            return feature_type.get_type_name()
        else:
            return str(feature_type)
    
    def _validate_features_exist(self, traj_data):
        """Check if all required features exist in traj_data."""
        for feature_key, selection_list in self.selections.items():
            if feature_key not in traj_data.features:
                raise ValueError(f"Feature '{feature_key}' not found in trajectory data")
            
            # Check if required data type (reduced/original) is available
            feature_data = traj_data.features[feature_key]
            for selection_dict in selection_list:
                use_reduced = selection_dict['use_reduced']
                if use_reduced:
                    if not hasattr(feature_data, 'reduced_data') or feature_data.reduced_data is None:
                        raise ValueError(f"Reduced data not available for feature '{feature_key}'. Run reduction first.")
                else:
                    if not hasattr(feature_data, 'data') or feature_data.data is None:
                        raise ValueError(f"Original data not available for feature '{feature_key}'.")
    
    def _initialize_selected_data_storage(self, traj_data, name: str):
        """Initialize storage for selections in traj_data."""
        if not hasattr(traj_data, 'selected_data'):
            traj_data.selected_data = {}
        traj_data.selected_data[name] = {}
    
    def _process_all_selections(self, traj_data, name: str):
        """Process all selections and store results."""
        for feature_key, selection_list in self.selections.items():
            all_column_indices = []
            use_reduced_indices = []  # Track which indices should use reduced data
            
            for selection_dict in selection_list:
                selection_string = selection_dict['selection']
                use_reduced = selection_dict['use_reduced']
                column_indices = self._get_column_indices_for_feature(
                    traj_data, feature_key, selection_string, use_reduced
                )
                
                # Store indices with their use_reduced flag
                start_idx = len(all_column_indices)
                all_column_indices.extend(column_indices)
                if use_reduced:
                    use_reduced_indices.extend(range(start_idx, len(all_column_indices)))
            
            # Remove duplicates while preserving use_reduced information
            unique_indices = []
            unique_use_reduced = []
            seen = set()
            
            for idx, col_idx in enumerate(all_column_indices):
                if col_idx not in seen:
                    seen.add(col_idx)
                    unique_indices.append(col_idx)
                    unique_use_reduced.append(idx in use_reduced_indices)
            
            # Store indices and their corresponding use_reduced flags
            traj_data.selected_data[name][feature_key] = {
                'indices': unique_indices,
                'use_reduced': unique_use_reduced  # Now a list of booleans
            }
    
    def _get_column_indices_for_feature(self, traj_data, feature_key: str, selection_string: str, use_reduced: bool) -> List[int]:
        """Get column indices for a specific feature selection."""
        feature_data = traj_data.features[feature_key]
        
        # Get the appropriate metadata based on use_reduced
        if use_reduced:
            feature_metadata = feature_data.reduced_feature_metadata
        else:
            feature_metadata = feature_data.feature_metadata
            
        if feature_metadata is None:
            raise ValueError(f"No feature metadata available for feature '{feature_key}'")
            
        if 'features' not in feature_metadata:
            raise ValueError(f"Invalid feature metadata structure for feature '{feature_key}'. Missing 'features' key.")
        
        return self._parse_selection(selection_string, feature_metadata)
    
    def _parse_selection(self, selection_string: str, feature_metadata: dict) -> List[int]:
        """
        Parse selection string and return matching column indices.
        
        Parameters:
        -----------
        selection_string : str
            Selection criteria (e.g., "res ALA", "resid 123-140", "all")
        feature_metadata : dict
            Feature metadata dictionary
            
        Returns:
        --------
        List[int]
            List of column indices matching the selection
        """
        if selection_string.strip().lower() == "all":
            return list(range(len(feature_metadata['features'])))
        
        return FeatureSelectorHelper.parse_selection(selection_string, feature_metadata['features'])
