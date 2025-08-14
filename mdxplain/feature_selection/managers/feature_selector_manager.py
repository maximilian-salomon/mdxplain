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

from typing import List, Union

from ..entities.feature_selector_data import FeatureSelectorData
from ..helpers.feature_selector_parse_core_helper import FeatureSelectorParseCoreHelper
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

    def __init__(self):
        """
        Initialize feature selector manager.

        Returns:
        --------
        None
            Initializes FeatureSelectorManager with empty configuration
        """
        pass

    def create(self, pipeline_data, name: str) -> None:
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
        pipeline_data,
        name: str,
        feature_type: Union[str, object],
        selection: str = "all",
        use_reduced: bool = False,
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
            Selection criteria string (e.g., "res ALA", "resid 123-140", "all")
        use_reduced : bool, default=False
            Whether to use reduced data (True) or original data (False)

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

        selector_data.add_selection(feature_key, selection, use_reduced)

        print(
            f"Added to selector '{name}': {feature_key} -> '{selection}' "
            f"(use_reduced={use_reduced})"
        )

    def select(self, pipeline_data, name: str) -> None:
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

        Standalone mode:
        >>> pipeline_data = PipelineData()
        >>> manager = FeatureSelectorManager()
        >>> manager.select(pipeline_data, "my_analysis")  # pipeline_data required

        Parameters:
        -----------
        pipeline_data : PipelineData
            Pipeline data object containing features and selector configurations
        name : str
            Name of the feature selector configuration to apply

        Returns:
        --------
        None
            Applies selections and stores results in pipeline_data.selected_data

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

        self._validate_features_exist(pipeline_data, selector_data)
        self._process_all_selections(pipeline_data, selector_data)

        print(f"Applied feature selector '{name}' successfully")

    def list_selectors(self, pipeline_data) -> List[str]:
        """
        Get list of all configured feature selector names.

        Warning:
        --------
        When using PipelineManager, do NOT provide the pipeline_data parameter.
        The PipelineManager automatically injects this parameter.

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

    def get_selector_summary(self, pipeline_data, name: str) -> dict:
        """
        Get summary information about a named selector configuration.

        Warning:
        --------
        When using PipelineManager, do NOT provide the pipeline_data parameter.
        The PipelineManager automatically injects this parameter.

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

    def remove_selector(self, pipeline_data, name: str) -> None:
        """
        Remove a named selector configuration.

        Warning:
        --------
        When using PipelineManager, do NOT provide the pipeline_data parameter.
        The PipelineManager automatically injects this parameter.

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

        # Note: Results are now stored directly in the FeatureSelectorData object
        # and will be removed automatically with the object

        print(f"Removed feature selector: '{name}'")

    def _validate_selector_exists(self, pipeline_data, name: str) -> None:
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
        self, pipeline_data, selector_data: FeatureSelectorData
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

    def _validate_feature_exists(self, feature_key: str, pipeline_data) -> None:
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
        self, feature_key: str, selections: List[dict], pipeline_data
    ) -> None:
        """
        Validate that required data types are available for a feature.

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
            If required data types are not available
        """
        feature_data = pipeline_data.feature_data[feature_key]

        for selection_dict in selections:
            use_reduced = selection_dict["use_reduced"]
            if use_reduced:
                if (
                    not hasattr(feature_data, "reduced_data")
                    or feature_data.reduced_data is None
                ):
                    raise ValueError(
                        f"Reduced data not available for feature '{feature_key}'. "
                        "Run reduction first."
                    )
            else:
                if not hasattr(feature_data, "data") or feature_data.data is None:
                    raise ValueError(
                        f"Original data not available for feature '{feature_key}'."
                    )

    def _process_all_selections(
        self, pipeline_data, selector_data: FeatureSelectorData
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

    def _process_feature_selections(
        self, pipeline_data, feature_key: str, selections: List[dict]
    ) -> dict:
        """
        Process all selections for a single feature.

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
            Dictionary with indices and use_reduced flags
        """
        all_column_indices, use_reduced_indices = self._collect_indices_for_feature(
            pipeline_data, feature_key, selections
        )

        unique_indices, unique_use_reduced = self._remove_duplicate_indices(
            all_column_indices, use_reduced_indices
        )

        return {
            "indices": unique_indices,
            "use_reduced": unique_use_reduced,
        }

    def _collect_indices_for_feature(
        self, pipeline_data, feature_key: str, selections: List[dict]
    ) -> tuple:
        """
        Collect all indices for a feature from all selections.

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
        tuple
            Tuple of (all_column_indices, use_reduced_indices)
        """
        all_column_indices: List[int] = []
        use_reduced_indices: List[int] = []

        for selection_dict in selections:
            indices = self._get_column_indices_for_feature(
                pipeline_data,
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
                unique_use_reduced.append(idx in use_reduced_indices)

        return unique_indices, unique_use_reduced

    def _get_column_indices_for_feature(
        self, pipeline_data, feature_key: str, selection_string: str, use_reduced: bool
    ) -> List[int]:
        """
        Get column indices for a specific feature selection.

        Parameters:
        -----------
        pipeline_data : PipelineData
            Pipeline data object
        feature_key : str
            Feature key
        selection_string : str
            Selection criteria string
        use_reduced : bool
            Whether to use reduced data

        Returns:
        --------
        List[int]
            List of matching column indices
        """
        feature_metadata = self._get_appropriate_metadata(
            pipeline_data, feature_key, use_reduced
        )
        self._validate_metadata_structure(feature_metadata, feature_key)
        return self._parse_selection(selection_string, feature_metadata)

    def _get_appropriate_metadata(
        self, pipeline_data, feature_key: str, use_reduced: bool
    ):
        """
        Get the appropriate metadata based on use_reduced flag.

        Parameters:
        -----------
        pipeline_data : PipelineData
            Pipeline data object
        feature_key : str
            Feature key
        use_reduced : bool
            Whether to use reduced metadata

        Returns:
        --------
        dict
            Feature metadata dictionary
        """
        feature_data = pipeline_data.feature_data[feature_key]

        if use_reduced:
            return feature_data.reduced_feature_metadata
        else:
            return feature_data.feature_metadata

    def _validate_metadata_structure(self, feature_metadata, feature_key: str) -> None:
        """
        Validate that metadata has the required structure.

        Parameters:
        -----------
        feature_metadata : dict or None
            Feature metadata dictionary
        feature_key : str
            Feature key for error messages

        Raises:
        -------
        ValueError
            If metadata is invalid
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
            Selection criteria string
        feature_metadata : dict
            Feature metadata dictionary

        Returns:
        --------
        List[int]
            List of matching column indices
        """
        if selection_string.strip().lower() == "all":
            return list(range(len(feature_metadata["features"])))

        return FeatureSelectorParseCoreHelper.parse_selection(
            selection_string, feature_metadata["features"]
        )
