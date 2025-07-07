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

    def add(
        self,
        feature_type: Union[str, object],
        selection: str = "all",
        use_reduced: bool = False,
    ):
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

        Returns:
        --------
        None
            Adds the feature type with selection criteria to the selector

        Raises:
        -------
        ValueError
            If feature_type is not a string or object with get_type_name method
        ValueError
            If feature_type is not found in traj_data.features
        ValueError
            If selection is not a string
        ValueError
            If selection is invalid
        ValueError
            If use_reduced is not a boolean
        ValueError
            If feature_type is not found in traj_data.features
        ValueError
            If selection is not a string
        Examples:
        ---------
        >>> selector = FeatureSelector()
        >>> selector.add("distances", "res ALA")
        >>> selector.add("distances", "resid 123-140", use_reduced=True)
        >>> selector.add("contacts", "all")
        """
        feature_key = self._convert_feature_type_to_key(feature_type)

        if feature_key not in self.selections:
            self.selections[feature_key] = []

        self.selections[feature_key].append(
            {"selection": selection, "use_reduced": use_reduced}
        )

    def select(self, traj_data, name: str):
        """
        Apply selections and store results in trajectory data.

        Parameters:
        -----------
        traj_data : TrajectoryData
            Trajectory data object containing computed features
        name : str
            Name to store the selection under

        Returns:
        --------
        None
            Applies selections and stores results in trajectory data

        Raises:
        -------
        ValueError
            If feature_type is not found in traj_data.features
        ValueError
            If selection is not a string
        ValueError
            If selection is invalid
        ValueError
            If feature_type is not found in traj_data.features
        ValueError
            If selection is not a string

        Examples:
        ---------
        >>> selector = FeatureSelector()
        >>> selector.add("distances", "res ALA")
        >>> selector.select(traj_data, "ala_analysis")
        >>>
        >>> # Access selected data
        >>> matrix = traj_data.get_selected_matrix("ala_analysis")
        >>> metadata = traj_data.get_selected_feature_metadata("ala_analysis")
        """
        self._validate_features_exist(traj_data)
        self._initialize_selected_data_storage(traj_data, name)
        self._process_all_selections(traj_data, name)

    def _convert_feature_type_to_key(self, feature_type: Union[str, object]) -> str:
        """
        Convert feature_type to string key.

        Parameters:
        -----------
        feature_type : str or object
            Feature type to convert

        Returns:
        --------
        str
            String representation of the feature type
        """
        if isinstance(feature_type, str):
            return feature_type
        elif hasattr(feature_type, "get_type_name"):
            return feature_type.get_type_name()
        else:
            return str(feature_type)

    def _validate_features_exist(self, traj_data):
        """
        Check if all required features exist in traj_data.

        Parameters:
        -----------
        traj_data : TrajectoryData
            Trajectory data object to validate

        Returns:
        --------
        None
            Validates that all required features exist in traj_data

        Raises:
        -------
        ValueError
            If feature_type is not found in traj_data.features
        ValueError
            If selection is not a string
        ValueError
            If selection is invalid
        """
        for feature_key, selection_list in self.selections.items():
            self._validate_feature_exists(feature_key, traj_data)
            self._validate_data_types_available(feature_key, selection_list, traj_data)

    def _validate_feature_exists(self, feature_key: str, traj_data):
        """
        Validate that a feature exists in trajectory data.

        Parameters:
        -----------
        feature_key : str
            Feature key to validate
        traj_data : TrajectoryData
            Trajectory data object

        Returns:
        --------
        None
            Validates that a feature exists in trajectory data

        Raises:
        -------
        ValueError
            If feature_key is not found in traj_data.features
        """
        if feature_key not in traj_data.features:
            raise ValueError(f"Feature '{feature_key}' not found in trajectory data")

    def _validate_data_types_available(
        self, feature_key: str, selection_list: list, traj_data
    ):
        """
        Validate that required data types are available for a feature.

        Parameters:
        -----------
        feature_key : str
            Feature key to validate
        selection_list : list
            List of selection dictionaries
        traj_data : TrajectoryData
            Trajectory data object

        Returns:
        --------
        None
            Validates that required data types are available for a feature

        Raises:
        -------
        ValueError
            If reduced data is not available for a feature
        ValueError
            If original data is not available for a feature
        """
        feature_data = traj_data.features[feature_key]

        for selection_dict in selection_list:
            use_reduced = selection_dict["use_reduced"]
            self._validate_single_data_type(feature_data, feature_key, use_reduced)

    def _validate_single_data_type(
        self, feature_data, feature_key: str, use_reduced: bool
    ):
        """
        Validate that a specific data type is available.

        Parameters:
        -----------
        feature_data : FeatureData
            Feature data object
        feature_key : str
            Feature key for error messages
        use_reduced : bool
            Whether reduced data is required

        Returns:
        --------
        None
            Validates that a specific data type is available

        Raises:
        -------
        ValueError
            If reduced data is not available for a feature
        ValueError
            If original data is not available for a feature
        """
        if use_reduced:
            self._validate_reduced_data_available(feature_data, feature_key)
        else:
            self._validate_original_data_available(feature_data, feature_key)

    def _validate_reduced_data_available(self, feature_data, feature_key: str):
        """
        Validate that reduced data is available.

        Parameters:
        -----------
        feature_data : FeatureData
            Feature data object
        feature_key : str
            Feature key for error messages

        Returns:
        --------
        None
            Validates that reduced data is available

        Raises:
        -------
        ValueError
            If reduced data is not available for a feature
        """
        if (
            not hasattr(feature_data, "reduced_data")
            or feature_data.reduced_data is None
        ):
            raise ValueError(
                f"Reduced data not available for feature '{feature_key}'. Run reduction first."
            )

    def _validate_original_data_available(self, feature_data, feature_key: str):
        """
        Validate that original data is available.

        Parameters:
        -----------
        feature_data : FeatureData
            Feature data object
        feature_key : str
            Feature key for error messages

        Returns:
        --------
        None
            Validates that original data is available

        Raises:
        -------
        ValueError
            If original data is not available for a feature
        """
        if not hasattr(feature_data, "data") or feature_data.data is None:
            raise ValueError(
                f"Original data not available for feature '{feature_key}'."
            )

    def _initialize_selected_data_storage(self, traj_data, name: str):
        """
        Initialize storage for selections in traj_data.

        Parameters:
        -----------
        traj_data : TrajectoryData
            Trajectory data object
        name : str
            Selection name

        Returns:
        --------
        None
            Initializes storage for selections in traj_data
        """
        if not hasattr(traj_data, "selected_data"):
            traj_data.selected_data = {}
        traj_data.selected_data[name] = {}

    def _process_all_selections(self, traj_data, name: str):
        """
        Process all selections and store results.

        Parameters:
        -----------
        traj_data : TrajectoryData
            Trajectory data object
        name : str
            Selection name

        Returns:
        --------
        None
            Processes all selections and stores results in traj_data

        Raises:
        -------
        ValueError
            If feature_key is not found in traj_data.features
        ValueError
            If selection is not a string
        """
        for feature_key, selection_list in self.selections.items():
            processed_indices = self._process_feature_selections(
                traj_data, feature_key, selection_list
            )
            traj_data.selected_data[name][feature_key] = processed_indices

    def _process_feature_selections(
        self, traj_data, feature_key: str, selection_list: list
    ) -> dict:
        """
        Process all selections for a single feature.

        Parameters:
        -----------
        traj_data : TrajectoryData
            Trajectory data object
        feature_key : str
            Feature key
        selection_list : list
            List of selection dictionaries

        Returns:
        --------
        dict
            Dictionary with indices and use_reduced flags

        Raises:
        -------
        ValueError
            If feature_key is not found in traj_data.features
        ValueError
            If selection is not a string
        """
        all_column_indices, use_reduced_indices = self._collect_indices_for_feature(
            traj_data, feature_key, selection_list
        )

        unique_indices, unique_use_reduced = self._remove_duplicate_indices(
            all_column_indices, use_reduced_indices
        )

        return {
            "indices": unique_indices,
            "use_reduced": unique_use_reduced,
        }

    def _collect_indices_for_feature(
        self, traj_data, feature_key: str, selection_list: list
    ) -> tuple:
        """
        Collect all indices for a feature from all selections.

        Parameters:
        -----------
        traj_data : TrajectoryData
            Trajectory data object containing features
        feature_key : str
            Feature key to collect indices for
        selection_list : list
            List of selection dictionaries with 'selection' and 'use_reduced' keys

        Returns:
        --------
        tuple
            Tuple of (all_column_indices, use_reduced_indices) where:
            - all_column_indices: List of all column indices from all selections
            - use_reduced_indices: List of indices indicating which columns use reduced data

        Raises:
        -------
        ValueError
            If feature_key is not found in traj_data.features
        ValueError
            If selection is not a string
        """
        all_column_indices: List[int] = []
        use_reduced_indices: List[int] = []

        for selection_dict in selection_list:
            indices = self._get_column_indices_for_feature(
                traj_data,
                feature_key,
                selection_dict["selection"],
                selection_dict["use_reduced"],
            )
            start_idx = len(all_column_indices)
            all_column_indices.extend(indices)

            if selection_dict["use_reduced"]:
                use_reduced_indices.extend(range(start_idx, len(all_column_indices)))

        return all_column_indices, use_reduced_indices

    def _remove_duplicate_indices(
        self, all_column_indices: list, use_reduced_indices: list
    ) -> tuple:
        """
        Remove duplicate indices while preserving use_reduced information.

        Parameters:
        -----------
        all_column_indices : list
            List of all column indices (may contain duplicates)
        use_reduced_indices : list
            List of indices indicating which positions in all_column_indices use reduced data

        Returns:
        --------
        tuple
            Tuple of (unique_indices, unique_use_reduced) where:
            - unique_indices: List of unique column indices
            - unique_use_reduced: List of boolean flags indicating use_reduced for each unique index

        Raises:
        -------
        ValueError
            If feature_key is not found in traj_data.features
        ValueError
            If selection is not a string
        """
        unique_indices = []
        unique_use_reduced = []
        seen = set()

        for idx, col_idx in enumerate(all_column_indices):
            if col_idx not in seen:
                seen.add(col_idx)
                unique_indices.append(col_idx)
                unique_use_reduced.append(idx in use_reduced_indices)

        return unique_indices, unique_use_reduced

    def _get_column_indices_for_feature(
        self, traj_data, feature_key: str, selection_string: str, use_reduced: bool
    ) -> List[int]:
        """
        Get column indices for a specific feature selection.

        Parameters:
        -----------
        traj_data : TrajectoryData
            Trajectory data object containing features
        feature_key : str
            Feature key to get indices for
        selection_string : str
            Selection criteria string (e.g., "res ALA", "resid 123-140", "all")
        use_reduced : bool
            Whether to use reduced data metadata or original data metadata

        Returns:
        --------
        List[int]
            List of column indices matching the selection criteria

        Raises:
        -------
        ValueError
            If feature_key is not found in traj_data.features
        ValueError
            If selection is not a string
        """
        feature_metadata = self._get_appropriate_metadata(
            traj_data, feature_key, use_reduced
        )
        self._validate_metadata_structure(feature_metadata, feature_key)
        return self._parse_selection(selection_string, feature_metadata)

    def _get_appropriate_metadata(self, traj_data, feature_key: str, use_reduced: bool):
        """
        Get the appropriate metadata based on use_reduced flag.

        Parameters:
        -----------
        traj_data : TrajectoryData
            Trajectory data object containing features
        feature_key : str
            Feature key to get metadata for
        use_reduced : bool
            If True, returns reduced_feature_metadata; if False, returns feature_metadata

        Returns:
        --------
        dict
            Feature metadata dictionary (either reduced or original)

        Raises:
        -------
        ValueError
            If feature_key is not found in traj_data.features
        """
        feature_data = traj_data.features[feature_key]

        if use_reduced:
            return feature_data.reduced_feature_metadata
        else:
            return feature_data.feature_metadata

    def _validate_metadata_structure(self, feature_metadata, feature_key: str):
        """
        Validate that metadata has the required structure.

        Parameters:
        -----------
        feature_metadata : dict or None
            Feature metadata dictionary to validate
        feature_key : str
            Feature key for error messages

        Returns:
        --------
        None
            Raises ValueError if feature_metadata is None or missing 'features' key

        Raises:
        -------
        ValueError
            If feature_metadata is None or missing 'features' key
        """
        if feature_metadata is None:
            raise ValueError(
                f"No feature metadata available for feature '{feature_key}'"
            )

        if "features" not in feature_metadata:
            raise ValueError(
                f"Invalid feature metadata structure for feature '{feature_key}'. "
                f"Missing 'features' key."
            )

    def _parse_selection(
        self, selection_string: str, feature_metadata: dict
    ) -> List[int]:
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

        Raises:
        -------
        ValueError
            If selection_string is not a string
        ValueError
            If feature_metadata is not a dictionary
        """
        if selection_string.strip().lower() == "all":
            return list(range(len(feature_metadata["features"])))

        return FeatureSelectorHelper.parse_selection(
            selection_string, feature_metadata["features"]
        )
