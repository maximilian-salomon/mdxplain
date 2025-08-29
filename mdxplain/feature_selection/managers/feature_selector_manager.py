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

"""
Feature selector manager for creating custom feature data matrices.

This module provides the FeatureSelectorManager class that manages multiple
named feature selector configurations and applies them to create custom
feature data matrices from computed features.
"""
from __future__ import annotations

from typing import List, Union, Dict, Any, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from ...pipeline.entities.pipeline_data import PipelineData

from ...feature.entities.feature_data import FeatureData
from ..entities.feature_selector_data import FeatureSelectorData
from ..helpers.feature_selector_parse_core_helper import FeatureSelectorParseCoreHelper
from ..helpers.common_denominator_helper import CommonDenominatorHelper
from ...utils.data_utils import DataUtils


class FeatureSelectorManager:
    """
    Manager for creating and applying feature selector configurations.

    This manager creates, stores, and applies named feature selector configurations
    to generate custom feature data matrices. Each selector can combine multiple
    feature types with different selection criteria and data preferences.

    The manager follows the same pattern as other managers (ClusterManager,
    DecompositionManager) by storing named entity instances and providing
    methods to create, configure, and apply them.
    """

    def __init__(self) -> None:
        """
        Initialize feature selector manager.

        Returns:
        --------
        None
            Initializes FeatureSelectorManager with empty configuration
        """
        pass

    def create(self, pipeline_data: PipelineData, name: str) -> None:
        """
        Create a new feature selector configuration.

        Creates a new FeatureSelectorData instance with the given name and stores
        it in the pipeline data for later configuration and use.

        Warning:
        --------
        When using PipelineManager, do NOT provide the pipeline_data parameter.
        The PipelineManager automatically injects this parameter.

        Pipeline mode:
        >>> pipeline = PipelineManager()
        >>> pipeline.feature_selector.create("my_analysis")  # NO pipeline_data parameter

        Standalone mode:
        >>> pipeline_data = PipelineData()
        >>> manager = FeatureSelectorManager()
        >>> manager.create(pipeline_data, "my_analysis")  # pipeline_data required

        Parameters:
        -----------
        pipeline_data : PipelineData
            Pipeline data object to store the selector configuration
        name : str
            Name identifier for the feature selector configuration

        Returns:
        --------
        None
            Creates and stores new FeatureSelectorData in pipeline_data

        Raises:
        -------
        ValueError
            If selector with given name already exists

        Examples:
        ---------
        >>> manager = FeatureSelectorManager()
        >>> pipeline_data = PipelineData()
        >>> manager.create(pipeline_data, "protein_analysis")
        >>> print("protein_analysis" in pipeline_data.selected_feature_data)
        True
        """
        if name in pipeline_data.selected_feature_data:
            raise ValueError(f"Feature selector '{name}' already exists")

        pipeline_data.selected_feature_data[name] = FeatureSelectorData(name)
        print(f"Created feature selector: '{name}'")

    def add(
        self,
        pipeline_data: PipelineData,
        name: str,
        feature_type: Union[str, object],
        selection: str = "all",
        use_reduced: bool = False,
        common_denominator: bool = True,
        traj_selection: Union[int, str, List[Union[int, str]], "all"] = "all",
        require_all_partners: bool = False,
    ) -> None:
        """
        Add a feature selection to a named selector configuration.

        Adds selection criteria for a feature type to an existing selector
        configuration. Multiple selections can be added for the same feature
        type with different criteria.

        Warning:
        --------
        When using PipelineManager, do NOT provide the pipeline_data parameter.
        The PipelineManager automatically injects this parameter.

        Pipeline mode:
        >>> pipeline = PipelineManager()
        >>> pipeline.feature_selector.add("analysis", "distances", "res ALA")  # NO pipeline_data

        Standalone mode:
        >>> pipeline_data = PipelineData()
        >>> manager = FeatureSelectorManager()
        >>> manager.add(pipeline_data, "analysis", "distances", "res ALA")  # pipeline_data required

        Parameters:
        -----------
        pipeline_data : PipelineData
            Pipeline data object containing selector configurations
        name : str
            Name of the existing feature selector configuration
        feature_type : str or object
            Feature type to select from (e.g., "distances", "contacts")
        selection : str, default="all"
            Selection criteria string (e.g., "res ALA", "resid 123-140", "7x50-8x50", "all")
        use_reduced : bool, default=False
            Whether to use reduced data (True) or original data (False)
        common_denominator : bool, default=True
            Whether to find common features across trajectories in traj_selection.
            If True, only features present in ALL selected trajectories are included.
            If False, union of all features from selected trajectories is used.
        traj_selection : int, str, list, or "all", default="all"
            Selection of trajectories to process:
            - int: trajectory index
            - str: trajectory name, tag (prefixed with "tag:"), or "all"
            - list: list of indices/names/tags
            - "all": all trajectories (default)
        require_all_partners : bool, default=False
            For pairwise features, require all partners to be present in selection

        Returns:
        --------
        None
            Adds selection configuration to the named selector

        Raises:
        -------
        ValueError
            If selector with given name does not exist

        Examples:
        ---------
        >>> manager = FeatureSelectorManager()
        >>> pipeline_data = PipelineData()
        >>> manager.create(pipeline_data, "analysis")
        >>> manager.add(pipeline_data, "analysis", "distances", "res ALA")
        >>> manager.add(pipeline_data, "analysis", "contacts", "resid 120-140", use_reduced=True)
        """
        self._validate_selector_exists(pipeline_data, name)

        feature_key = DataUtils.get_type_key(feature_type)
        selector_data = pipeline_data.selected_feature_data[name]

        selector_data.add_selection(feature_key, selection, use_reduced, 
                                   common_denominator, traj_selection, require_all_partners)

        print(
            f"Added to selector '{name}': {feature_key} -> '{selection}' "
            f"(use_reduced={use_reduced}, common_denominator={common_denominator}, "
            f"traj_selection={traj_selection}, require_all_partners={require_all_partners})"
        )

    def select(self, pipeline_data: PipelineData, name: str, reference_traj: Union[int, str] = 0) -> None:
        """
        Apply a named selector configuration to create selected feature matrix.

        Applies all selection criteria from a named selector configuration to
        the computed features in pipeline_data and stores the results for later
        access through data access methods.

        Warning:
        --------
        When using PipelineManager, do NOT provide the pipeline_data parameter.
        The PipelineManager automatically injects this parameter.

        Pipeline mode:
        >>> pipeline = PipelineManager()
        >>> pipeline.feature_selector.select("my_analysis")  # NO pipeline_data parameter
        >>> # Or with specific reference trajectory
        >>> pipeline.feature_selector.select("my_analysis", reference_traj=2)

        Standalone mode:
        >>> pipeline_data = PipelineData()
        >>> manager = FeatureSelectorManager()
        >>> manager.select(pipeline_data, "my_analysis")  # pipeline_data required
        >>> # Or with specific reference trajectory
        >>> manager.select(pipeline_data, "my_analysis", reference_traj="system_A")

        Parameters:
        -----------
        pipeline_data : PipelineData
            Pipeline data object containing features and selector configurations
        name : str
            Name of the feature selector configuration to apply
        reference_traj : Union[int, str], default=0
            Trajectory index or name to use as reference for metadata extraction

        Returns:
        --------
        None
            Applies selections and stores results with reference trajectory info

        Raises:
        -------
        ValueError
            If selector with given name does not exist
        ValueError
            If required features are not computed
        ValueError
            If required data types (original/reduced) are not available

        Examples:
        ---------
        >>> manager = FeatureSelectorManager()
        >>> pipeline_data = PipelineData()
        >>> # ... configure selector and compute features ...
        >>> manager.select(pipeline_data, "protein_analysis")
        >>> # Results available via pipeline_data.selected_data["protein_analysis"]
        """
        self._validate_selector_exists(pipeline_data, name)

        selector_data = pipeline_data.selected_feature_data[name]

        if not selector_data.get_feature_keys():
            print(f"Warning: No selections configured for selector '{name}'")
            return

        reference_traj = pipeline_data.trajectory_data.get_trajectory_indices(reference_traj)[0]

        # Store reference trajectory in selector data
        selector_data.set_reference_trajectory(reference_traj)

        self._validate_features_exist(pipeline_data, selector_data)
        self._process_all_selections(pipeline_data, selector_data)

        print(f"Applied feature selector '{name}' with reference trajectory {reference_traj} successfully")

    def list_selectors(self, pipeline_data: PipelineData) -> List[str]:
        """
        Get list of all configured feature selector names.

        Warning:
        --------
        When using PipelineManager, do NOT provide the pipeline_data parameter.
        The PipelineManager automatically injects this parameter.

        Pipeline mode:
        >>> pipeline = PipelineManager()
        >>> pipeline.feature_selector.list_selectors()  # NO pipeline_data parameter

        Standalone mode:
        >>> pipeline_data = PipelineData()
        >>> manager = FeatureSelectorManager()
        >>> manager.list_selectors(pipeline_data)  # pipeline_data required

        Parameters:
        -----------
        pipeline_data : PipelineData
            Pipeline data object containing selector configurations

        Returns:
        --------
        List[str]
            List of configured feature selector names

        Examples:
        ---------
        >>> manager = FeatureSelectorManager()
        >>> pipeline_data = PipelineData()
        >>> manager.create(pipeline_data, "analysis1")
        >>> manager.create(pipeline_data, "analysis2")
        >>> selectors = manager.list_selectors(pipeline_data)
        >>> print(sorted(selectors))
        ['analysis1', 'analysis2']
        """
        if not hasattr(pipeline_data, "selected_feature_data"):
            return []
        return list(pipeline_data.selected_feature_data.keys())

    def get_selector_summary(self, pipeline_data: PipelineData, name: str) -> dict:
        """
        Get summary information about a named selector configuration.

        Warning:
        --------
        When using PipelineManager, do NOT provide the pipeline_data parameter.
        The PipelineManager automatically injects this parameter.

        Pipeline mode:
        >>> pipeline = PipelineManager()
        >>> pipeline.feature_selector.get_selector_summary("analysis")  # NO pipeline_data parameter

        Standalone mode:
        >>> pipeline_data = PipelineData()
        >>> manager = FeatureSelectorManager()
        >>> manager.get_selector_summary(pipeline_data, "analysis")  # pipeline_data required

        Parameters:
        -----------
        pipeline_data : PipelineData
            Pipeline data object containing selector configurations
        name : str
            Name of the feature selector configuration

        Returns:
        --------
        dict
            Summary information about the selector configuration

        Raises:
        -------
        ValueError
            If selector with given name does not exist

        Examples:
        ---------
        >>> manager = FeatureSelectorManager()
        >>> summary = manager.get_selector_summary(pipeline_data, "analysis")
        >>> print(summary['feature_count'])
        2
        """
        self._validate_selector_exists(pipeline_data, name)
        return pipeline_data.selected_feature_data[name].get_summary()

    def remove_selector(self, pipeline_data: PipelineData, name: str) -> None:
        """
        Remove a named selector configuration.

        Warning:
        --------
        When using PipelineManager, do NOT provide the pipeline_data parameter.
        The PipelineManager automatically injects this parameter.

        Pipeline mode:
        >>> pipeline = PipelineManager()
        >>> pipeline.feature_selector.remove_selector("old_analysis")  # NO pipeline_data parameter

        Standalone mode:
        >>> pipeline_data = PipelineData()
        >>> manager = FeatureSelectorManager()
        >>> manager.remove_selector(pipeline_data, "old_analysis")  # pipeline_data required

        Parameters:
        -----------
        pipeline_data : PipelineData
            Pipeline data object containing selector configurations
        name : str
            Name of the feature selector configuration to remove

        Returns:
        --------
        None
            Removes selector configuration and any associated selected data

        Raises:
        -------
        ValueError
            If selector with given name does not exist

        Examples:
        ---------
        >>> manager = FeatureSelectorManager()
        >>> manager.remove_selector(pipeline_data, "old_analysis")
        """
        self._validate_selector_exists(pipeline_data, name)
        del pipeline_data.selected_feature_data[name]
        print(f"Removed feature selector: '{name}'")

    def _validate_selector_exists(self, pipeline_data: PipelineData, name: str) -> None:
        """
        Validate that a named selector configuration exists.

        Parameters:
        -----------
        pipeline_data : PipelineData
            Pipeline data object to check
        name : str
            Name of selector to validate

        Raises:
        -------
        ValueError
            If selector does not exist
        """
        if (
            not hasattr(pipeline_data, "selected_feature_data")
            or name not in pipeline_data.selected_feature_data
        ):
            raise ValueError(f"Feature selector '{name}' does not exist")

    def _validate_features_exist(
        self, pipeline_data: PipelineData, selector_data: FeatureSelectorData
    ) -> None:
        """
        Validate that all required features exist in pipeline_data.

        Parameters:
        -----------
        pipeline_data : PipelineData
            Pipeline data object to validate
        selector_data : FeatureSelectorData
            Selector configuration to validate

        Raises:
        -------
        ValueError
            If required features or data types are not available
        """
        for feature_key in selector_data.get_feature_keys():
            self._validate_feature_exists(feature_key, pipeline_data)
            selections = selector_data.get_selections(feature_key)
            self._validate_data_types_available(feature_key, selections, pipeline_data)

    def _validate_feature_exists(self, feature_key: str, pipeline_data: PipelineData) -> None:
        """
        Validate that a feature exists in pipeline data.

        Parameters:
        -----------
        feature_key : str
            Feature key to validate
        pipeline_data : PipelineData
            Pipeline data object

        Raises:
        -------
        ValueError
            If feature does not exist
        """
        if feature_key not in pipeline_data.feature_data:
            raise ValueError(f"Feature '{feature_key}' not found in pipeline data")

    def _validate_data_types_available(
        self, feature_key: str, selections: List[Dict[str, Any]], pipeline_data: PipelineData
    ) -> None:
        """
        Validate that required data types are available for ALL relevant trajectories.

        Parameters:
        -----------
        feature_key : str
            Feature key to validate
        selections : List[dict]
            List of selection configurations
        pipeline_data : PipelineData
            Pipeline data object

        Raises:
        -------
        ValueError
            If required data types are not available for any relevant trajectory
        """
        feature_data_dict = pipeline_data.feature_data[feature_key]

        for selection_dict in selections:
            use_reduced = selection_dict["use_reduced"]
            traj_selection = selection_dict.get("traj_selection", "all")
            
            # Get trajectory indices and validate them
            traj_indices = pipeline_data.trajectory_data.get_trajectory_indices(traj_selection)
            missing_trajectories = self._check_trajectories_for_data_type(
                feature_data_dict, traj_indices, use_reduced
            )
            
            # Report errors with trajectory-specific information
            if missing_trajectories:
                self._raise_missing_data_error(feature_key, missing_trajectories, use_reduced)

    def _check_trajectories_for_data_type(
        self, feature_data_dict: dict, traj_indices: List[int], use_reduced: bool
    ) -> List[int]:
        """
        Check which trajectories are missing the required data type.

        Parameters:
        -----------
        feature_data_dict : dict
            Dictionary mapping trajectory indices to FeatureData objects
        traj_indices : List[int]
            Trajectory indices to check
        use_reduced : bool
            Whether to check for reduced data (True) or original data (False)

        Returns:
        --------
        List[int]
            List of trajectory indices that are missing the required data type
        """
        missing_trajectories = []
        
        for traj_idx in traj_indices:
            if traj_idx not in feature_data_dict:
                missing_trajectories.append(traj_idx)
                continue
                
            feature_data = feature_data_dict[traj_idx]
            
            if use_reduced:
                if (
                    not hasattr(feature_data, "reduced_data")
                    or feature_data.reduced_data is None
                ):
                    missing_trajectories.append(traj_idx)
            else:
                if not hasattr(feature_data, "data") or feature_data.data is None:
                    missing_trajectories.append(traj_idx)
        
        return missing_trajectories

    def _raise_missing_data_error(
        self, feature_key: str, missing_trajectories: List[int], use_reduced: bool
    ) -> None:
        """
        Raise error with specific information about missing data.

        Parameters:
        -----------
        feature_key : str
            Feature key that has missing data
        missing_trajectories : List[int]
            Trajectory indices that are missing the data
        use_reduced : bool
            Whether reduced or original data is missing

        Raises:
        -------
        ValueError
            Always raised with descriptive error message
        """
        data_type = "reduced" if use_reduced else "original"
        error_msg = (
            f"{data_type.capitalize()} data not available for feature '{feature_key}' "
            f"in trajectories {missing_trajectories}. "
            f"{'Run reduction first.' if use_reduced else 'Feature computation may have failed.'}"
        )
        raise ValueError(error_msg)

    def _process_all_selections(
        self, pipeline_data: PipelineData, selector_data: FeatureSelectorData
    ) -> None:
        """
        Process all selections and store results.

        Parameters:
        -----------
        pipeline_data : PipelineData
            Pipeline data object
        selector_data : FeatureSelectorData
            Selector configuration
        """
        for feature_key in selector_data.get_feature_keys():
            selections = selector_data.get_selections(feature_key)
            processed_indices = self._process_feature_selections(
                pipeline_data, feature_key, selections
            )
            selector_data.store_results(feature_key, processed_indices)
        
        # Calculate and store total number of columns
        total_columns = self._calculate_total_columns(selector_data)
        selector_data.set_n_columns(total_columns)

    def _process_feature_selections(
        self, pipeline_data: PipelineData, feature_key: str, selections: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Process all selections for a single feature per trajectory.
        
        Each selection is processed individually. If a selection has common_denominator=True,
        the CommonDenominatorHelper is applied only to that specific selection.
        All results are then combined using union.

        Parameters:
        -----------
        pipeline_data : PipelineData
            Pipeline data object
        feature_key : str
            Feature key
        selections : List[dict]
            List of selection configurations

        Returns:
        --------
        dict
            Dictionary with trajectory-specific indices and use_reduced flags:
            {"trajectory_indices": {traj_idx: {"indices": [...], "use_reduced": [...]}}}
        """
        all_trajectory_results = {}

        # Process each selection individually
        for selection_dict in selections:
            selection_trajectory_results = self._process_single_selection(
                pipeline_data, feature_key, selection_dict
            )
            
            # Merge results from this selection into overall results
            for traj_idx, result in selection_trajectory_results.items():
                if traj_idx not in all_trajectory_results:
                    all_trajectory_results[traj_idx] = {
                        "indices": [],
                        "use_reduced": [],
                    }
                
                # Add indices from this selection
                all_trajectory_results[traj_idx]["indices"].extend(result["indices"])
                all_trajectory_results[traj_idx]["use_reduced"].extend(result["use_reduced"])
        
        # Remove duplicates for each trajectory
        for traj_idx in all_trajectory_results:
            unique_indices, unique_use_reduced = self._remove_duplicate_indices(
                all_trajectory_results[traj_idx]["indices"],
                all_trajectory_results[traj_idx]["use_reduced"]
            )
            all_trajectory_results[traj_idx] = {
                "indices": unique_indices,
                "use_reduced": unique_use_reduced,
            }
        
        return {"trajectory_indices": all_trajectory_results}
    
    def _process_single_selection(
        self, pipeline_data: PipelineData, feature_key: str, selection_dict: Dict[str, Any]
    ) -> Dict[int, Any]:
        """
        Process a single selection and apply common_denominator if needed.

        Parameters:
        -----------
        pipeline_data : PipelineData
            Pipeline data object
        feature_key : str
            Feature key
        selection_dict : dict
            Single selection configuration

        Returns:
        --------
        dict
            Dictionary with trajectory-specific indices and use_reduced flags
        """
        trajectory_results = {}
        
        # Get all trajectories that have this feature
        feature_data_dict = pipeline_data.feature_data[feature_key]
        
        # Get trajectory selection for this specific selection
        traj_selection = selection_dict.get("traj_selection", "all")
        selected_traj_indices = pipeline_data.trajectory_data.get_trajectory_indices(traj_selection)
        
        # Process each trajectory
        for traj_idx in feature_data_dict.keys():
            # Skip if this selection doesn't apply to this trajectory
            if traj_idx not in selected_traj_indices:
                continue
                
            # Get indices for this trajectory and selection
            feature_data = feature_data_dict[traj_idx]
            selection_indices = self._get_indices_for_selection(
                feature_data, selection_dict, feature_key
            )
            
            if selection_indices:  # Only store if there are indices
                trajectory_results[traj_idx] = {
                    "indices": selection_indices,
                    "use_reduced": [selection_dict["use_reduced"]] * len(selection_indices),
                }
        
        # Apply common denominator filtering ONLY if this selection requires it
        if selection_dict.get("common_denominator", False):
            trajectory_results = CommonDenominatorHelper.apply_common_denominator(
                pipeline_data, feature_key, trajectory_results
            )
        
        return trajectory_results
    
    def _calculate_total_columns(self, selector_data: FeatureSelectorData) -> int:
        """
        Calculate total number of columns in final matrix.
        
        Validates that all trajectories have the same number of columns for each feature
        to ensure consistent matrix construction.
        
        Parameters:
        -----------
        selector_data : FeatureSelectorData
            Selector data with stored results
            
        Returns:
        --------
        int
            Total number of columns across all features and trajectories
            
        Raises:
        -------
        ValueError
            If trajectories have inconsistent column counts for any feature
        """
        total_columns = 0
        all_results = selector_data.get_all_results()
        
        for feature_type, selection_info in all_results.items():
            trajectory_indices_data = selection_info.get("trajectory_indices", {})
            
            if not trajectory_indices_data:
                continue
            
            
            # Check that all trajectories have same number of columns
            column_counts = []
            for traj_idx, traj_data in trajectory_indices_data.items():
                n_cols = len(traj_data.get("indices", []))
                column_counts.append((traj_idx, n_cols))
            
            if not column_counts:
                continue
            
            # Validate consistency across trajectories
            first_count = column_counts[0][1]
            inconsistent = [(idx, count) for idx, count in column_counts if count != first_count]
            
            if inconsistent:
                raise ValueError(
                    f"Feature '{feature_type}' has inconsistent column counts across trajectories. "
                    f"Expected {first_count} columns, but found: {inconsistent}. "
                    f"Use common_denominator=True or ensure all trajectories have same features."
                )
            
            total_columns += first_count
        
        return total_columns

    def _collect_indices_for_trajectory(
        self, pipeline_data: PipelineData, feature_key: str, selections: List[Dict[str, Any]], traj_idx: int
    ) -> Tuple[List[int], List[bool]]:
        """
        Collect indices for a specific trajectory from all selections.

        Parameters:
        -----------
        pipeline_data : PipelineData
            Pipeline data object
        feature_key : str
            Feature key
        selections : List[dict]
            List of selection configurations
        traj_idx : int
            Trajectory index to collect indices for

        Returns:
        --------
        tuple
            Tuple of (indices, use_reduced_flags) for this trajectory
        """
        trajectory_indices = []
        trajectory_use_reduced = []
        
        feature_data = pipeline_data.feature_data[feature_key][traj_idx]
        
        for selection_dict in selections:
            traj_selection = selection_dict.get("traj_selection", "all")
            
            # Check if this selection applies to this trajectory
            selected_traj_indices = pipeline_data.trajectory_data.get_trajectory_indices(traj_selection)
            if traj_idx not in selected_traj_indices:
                continue  # Skip this selection for this trajectory
            
            # Get indices for this trajectory
            selection_indices = self._get_indices_for_selection(
                feature_data, selection_dict, feature_key
            )
            
            # Add to trajectory results
            for idx in selection_indices:
                trajectory_indices.append(idx)
                trajectory_use_reduced.append(selection_dict["use_reduced"])
        
        return trajectory_indices, trajectory_use_reduced

    def _get_indices_for_selection(
        self, feature_data: FeatureData, selection_dict: Dict[str, Any], feature_key: str
    ) -> List[int]:
        """
        Get indices for a selection on a specific trajectory's feature data.

        Parameters:
        -----------
        feature_data : FeatureData
            Single trajectory's feature data
        selection_dict : dict
            Selection configuration
        feature_key : str
            Feature key for error messages

        Returns:
        --------
        List[int]
            List of matching indices for this trajectory
        """
        use_reduced = selection_dict["use_reduced"]
        selection_string = selection_dict["selection"]
        
        # Get appropriate metadata for this trajectory
        if use_reduced:
            if feature_data.reduced_feature_metadata is not None:
                metadata = feature_data.reduced_feature_metadata
            else:
                raise ValueError(
                    f"Reduced metadata not available for feature '{feature_key}'. "
                    "Run reduction first."
                )
        else:
            if feature_data.feature_metadata is not None:
                metadata = feature_data.feature_metadata
            else:
                raise ValueError(
                    f"Original metadata not available for feature '{feature_key}'."
                )
        
        # Find matching indices in this trajectory's metadata       
        features_list = metadata.get("features", [])
        require_all_partners = selection_dict.get("require_all_partners", False)
        matching_indices = FeatureSelectorParseCoreHelper.parse_selection(
            selection_string, features_list, require_all_partners
        )
        
        return matching_indices


    def _remove_duplicate_indices(
        self, all_column_indices: list, use_reduced_indices: list
    ) -> tuple:
        """
        Remove duplicate indices while preserving use_reduced information.

        Parameters:
        -----------
        all_column_indices : list
            List of all column indices
        use_reduced_indices : list
            List of indices that use reduced data

        Returns:
        --------
        tuple
            Tuple of (unique_indices, unique_use_reduced)
        """
        unique_indices = []
        unique_use_reduced = []
        seen = set()

        for idx, col_idx in enumerate(all_column_indices):
            if col_idx not in seen:
                seen.add(col_idx)
                unique_indices.append(col_idx)
                unique_use_reduced.append(use_reduced_indices[idx])

        # Sort by indices for consistent, interpretable order
        sorted_pairs = sorted(zip(unique_indices, unique_use_reduced))
        sorted_indices = [idx for idx, _ in sorted_pairs]
        sorted_use_reduced = [reduced for _, reduced in sorted_pairs]

        return sorted_indices, sorted_use_reduced

    def save(self, pipeline_data: PipelineData, save_path: str) -> None:
        """
        Save all feature selector data to single file.

        Warning:
        --------
        When using PipelineManager, do NOT provide the pipeline_data parameter.
        The PipelineManager automatically injects this parameter.

        Pipeline mode:
        >>> pipeline = PipelineManager()
        >>> pipeline.feature_selector.save('feature_selector.npy')  # NO pipeline_data parameter

        Standalone mode:
        >>> pipeline_data = PipelineData()
        >>> manager = FeatureSelectorManager()
        >>> manager.save(pipeline_data, 'feature_selector.npy')  # pipeline_data required

        Parameters:
        -----------
        pipeline_data : PipelineData
            Pipeline data container with feature selector data
        save_path : str
            Path where to save all feature selector data in one file

        Returns:
        --------
        None
            Saves all feature selector data to the specified file
            
        Examples:
        ---------
        >>> manager.save(pipeline_data, 'feature_selector.npy')
        """
        DataUtils.save_object(pipeline_data.feature_selector_data, save_path)

    def load(self, pipeline_data: PipelineData, load_path: str) -> None:
        """
        Load all feature selector data from single file.

        Warning:
        --------
        When using PipelineManager, do NOT provide the pipeline_data parameter.
        The PipelineManager automatically injects this parameter.

        Pipeline mode:
        >>> pipeline = PipelineManager()
        >>> pipeline.feature_selector.load('feature_selector.npy')  # NO pipeline_data parameter

        Standalone mode:
        >>> pipeline_data = PipelineData()
        >>> manager = FeatureSelectorManager()
        >>> manager.load(pipeline_data, 'feature_selector.npy')  # pipeline_data required

        Parameters:
        -----------
        pipeline_data : PipelineData
            Pipeline data container to load feature selector data into
        load_path : str
            Path to saved feature selector data file

        Returns:
        --------
        None
            Loads all feature selector data from the specified file
            
        Examples:
        ---------
        >>> manager.load(pipeline_data, 'feature_selector.npy')
        """
        temp_dict = {}
        DataUtils.load_object(temp_dict, load_path)
        pipeline_data.feature_selector_data = temp_dict

    def print_info(self, pipeline_data: PipelineData) -> None:
        """
        Print featureselectordata information.

        Warning:
        --------
        When using PipelineManager, do NOT provide the pipeline_data parameter.
        The PipelineManager automatically injects this parameter.

        Pipeline mode:
        >>> pipeline = PipelineManager()
        >>> pipeline.feature_selector.print_info()  # NO pipeline_data parameter

        Standalone mode:
        >>> pipeline_data = PipelineData()
        >>> manager = FeatureSelectorManager()
        >>> manager.print_info(pipeline_data)  # pipeline_data required

        Parameters:
        -----------
        pipeline_data : PipelineData
            Pipeline data container with featureselectordata

        Returns:
        --------
            Prints featureselectordata information to console

        Examples:
        ---------
        >>> manager.print_info(pipeline_data)
        """
        if len(pipeline_data.feature_selector_data) == 0:
            print("No featureselectordata data available.")
            return

        print("=== FeatureSelectorData Information ===")
        data_names = list(pipeline_data.feature_selector_data.keys())
        print(f"FeatureSelectorData Names: {len(data_names)} ({", ".join(data_names)})")
        
        for name, data in pipeline_data.feature_selector_data.items():
            print(f"\n--- {name} ---")
            data.print_info()
