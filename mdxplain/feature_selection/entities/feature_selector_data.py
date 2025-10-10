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
Feature selector data entity for storing selection configurations.

This module contains the FeatureSelectorData class that stores feature
selection configurations including selection criteria and data type preferences.
"""

from typing import Dict, List, Optional, Union

from ...utils.data_utils import DataUtils


class FeatureSelectorData:
    """
    Data entity for storing feature selector configurations.

    Stores selection criteria for different feature types, allowing multiple
    selections per feature type with different data preferences (original vs reduced).

    This entity follows the same pattern as ClusterData and DecompositionData,
    providing a container for feature selector configurations and results that can be
    stored and retrieved by name.

    Attributes
    ----------
    name : str
        Name identifier for this feature selector configuration
    selections : Dict[str, List[dict]]
        Dictionary mapping feature type keys to lists of selection dictionaries.
        Each selection dict contains:

        - 'selection': Selection criteria string (e.g., "res ALA", "resid 123-140")
        - 'use_reduced': Boolean flag for data type preference
    selection_results : Dict[str, dict]
        Dictionary mapping feature type keys to computed selection results.
        Each result dict contains:
        
        - 'indices': List of selected column indices
        - 'use_reduced': List of boolean flags for reduced data usage
    reference_trajectory : int, optional
        Trajectory index to use as reference for metadata extraction

    Examples
    --------
    >>> selector_data = FeatureSelectorData("my_analysis")
    >>> selector_data.add_selection("distances", "res ALA", use_reduced=False)
    >>> selector_data.add_selection("contacts", "resid 120-140", use_reduced=True)
    """

    def __init__(self, name: str):
        """
        Initialize feature selector data with given name.

        Parameters
        ----------
        name : str
            Name identifier for this feature selector configuration

        Returns
        -------
        None
            Initializes empty FeatureSelectorData with given name

        Examples
        --------
        >>> selector_data = FeatureSelectorData("protein_analysis")
        >>> print(selector_data.name)
        'protein_analysis'
        """
        self.name = name
        self.selections: Dict[str, List[dict]] = {}
        self.selection_results: Dict[str, dict] = {}
        self.reference_trajectory: Optional[int] = None
        self.n_columns: Optional[int] = None

    def add_selection(
        self, feature_key: str, selection: str, use_reduced: bool = False,
        common_denominator: bool = True, traj_selection: Union[int, str, List[Union[int, str]]] = "all", require_all_partners: bool = False
    ) -> None:
        """
        Add a selection configuration for a feature type.

        Parameters
        ----------
        feature_key : str
            Feature type key (e.g., "distances", "contacts")
        selection : str
            Selection criteria string (e.g., "res ALA", "resid 123-140", "7x50-8x50", "all")
        use_reduced : bool, default=False
            Whether to use reduced data (True) or original data (False)
        common_denominator : bool, default=True
            Whether to find common features across trajectories in traj_selection
        traj_selection : int, str, list, or "all", default="all"
            Selection of trajectories to process
        require_all_partners : bool, default=False
            For pairwise features, require all partners to be present in selection

        Returns
        -------
        None
            Adds selection configuration to the selections dictionary

        Examples
        --------
        >>> selector_data = FeatureSelectorData("analysis")
        >>> selector_data.add_selection("distances", "res ALA")
        >>> selector_data.add_selection("contacts", "resid 120-140", use_reduced=True)
        >>> print(len(selector_data.selections["distances"]))
        1
        """
        if feature_key not in self.selections:
            self.selections[feature_key] = []

        self.selections[feature_key].append({
            "selection": selection, 
            "use_reduced": use_reduced,
            "common_denominator": common_denominator,
            "traj_selection": traj_selection,
            "require_all_partners": require_all_partners
        })

    def get_selections(self, feature_key: str) -> List[dict]:
        """
        Get all selection configurations for a feature type.

        Parameters
        ----------
        feature_key : str
            Feature type key to get selections for

        Returns
        -------
        List[dict]
            List of selection dictionaries for the feature type.
            Returns empty list if feature_key not found.

        Examples
        --------
        >>> selector_data = FeatureSelectorData("analysis")
        >>> selector_data.add_selection("distances", "res ALA")
        >>> selector_data.add_selection("distances", "res HIS")
        >>> selections = selector_data.get_selections("distances")
        >>> print(len(selections))
        2
        """
        return self.selections.get(feature_key, [])

    def has_feature(self, feature_key: str) -> bool:
        """
        Check if feature type has any selections configured.

        Parameters
        ----------
        feature_key : str
            Feature type key to check

        Returns
        -------
        bool
            True if feature type has selections, False otherwise

        Examples
        --------
        >>> selector_data = FeatureSelectorData("analysis")
        >>> selector_data.add_selection("distances", "res ALA")
        >>> print(selector_data.has_feature("distances"))
        True
        >>> print(selector_data.has_feature("contacts"))
        False
        """
        return feature_key in self.selections and len(self.selections[feature_key]) > 0

    def get_feature_keys(self) -> List[str]:
        """
        Get list of all configured feature type keys.

        Returns
        -------
        List[str]
            List of feature type keys that have selections configured

        Examples
        --------
        >>> selector_data = FeatureSelectorData("analysis")
        >>> selector_data.add_selection("distances", "res ALA")
        >>> selector_data.add_selection("contacts", "resid 120")
        >>> keys = selector_data.get_feature_keys()
        >>> print(sorted(keys))
        ['contacts', 'distances']
        """
        return [key for key, selections in self.selections.items() if selections]

    def clear_selections(self, feature_key: str = None) -> None:
        """
        Clear selections for a feature type or all selections.

        Parameters
        ----------
        feature_key : str, optional
            Feature type key to clear selections for.
            If None, clears all selections.

        Returns
        -------
        None
            Clears specified or all selections

        Examples
        --------
        >>> selector_data = FeatureSelectorData("analysis")
        >>> selector_data.add_selection("distances", "res ALA")
        >>> selector_data.add_selection("contacts", "resid 120")
        >>> selector_data.clear_selections("distances")  # Clear only distances
        >>> print(selector_data.has_feature("distances"))
        False
        >>> selector_data.clear_selections()  # Clear all
        >>> print(len(selector_data.get_feature_keys()))
        0
        """
        if feature_key is None:
            self.selections.clear()
        elif feature_key in self.selections:
            del self.selections[feature_key]

    def get_summary(self) -> dict:
        """
        Get summary information about the selector configuration.

        Returns
        -------
        dict
            Summary dictionary with configuration statistics

        Examples
        --------
        >>> selector_data = FeatureSelectorData("analysis")
        >>> selector_data.add_selection("distances", "res ALA")
        >>> selector_data.add_selection("contacts", "resid 120", use_reduced=True)
        >>> summary = selector_data.get_summary()
        >>> print(summary['name'])
        'analysis'
        >>> print(summary['feature_count'])
        2
        """
        total_selections = sum(
            len(selections) for selections in self.selections.values()
        )
        reduced_selections = sum(
            len([s for s in selections if s.get("use_reduced", False)])
            for selections in self.selections.values()
        )

        return {
            "name": self.name,
            "feature_count": len(self.get_feature_keys()),
            "total_selections": total_selections,
            "reduced_selections": reduced_selections,
            "original_selections": total_selections - reduced_selections,
        }

    def store_results(self, feature_key: str, result_data: dict) -> None:
        """
        Store selection results for a feature type.

        Stores the computed indices and metadata for a feature type selection.
        This method is called by FeatureSelectorManager after processing selections.

        Parameters
        ----------
        feature_key : str
            Feature type key to store results for
        result_data : dict
            Result dictionary containing:
            
            - 'indices': List of selected column indices
            - 'use_reduced': List of boolean flags for reduced data usage

        Returns
        -------
        None
            Stores results in selection_results dictionary

        Examples
        --------
        >>> selector_data = FeatureSelectorData("analysis")
        >>> results = {"trajectory_indices": {traj_idx: {"indices": [...], "use_reduced": [...]}}}
        >>> selector_data.store_results("distances", results)
        >>> print(selector_data.has_results("distances"))
        True
        """
        self.selection_results[feature_key] = result_data

    def get_results(self, feature_key: str) -> dict:
        """
        Get selection results for a feature type.

        Retrieves the stored selection results (indices and flags) for a
        specific feature type. Returns empty dict if no results found.

        Parameters
        ----------
        feature_key : str
            Feature type key to get results for

        Returns
        -------
        dict
            Result dictionary with 'indices' and 'use_reduced' keys,
            or empty dict if no results stored

        Examples
        --------
        >>> selector_data = FeatureSelectorData("analysis")
        >>> results = selector_data.get_results("distances")
        >>> if results:
        ...     print(f"Selected {len(results['indices'])} features")
        """
        return self.selection_results.get(feature_key, {})

    def get_all_results(self) -> Dict[str, dict]:
        """
        Get all stored selection results.

        Returns the complete selection_results dictionary containing
        results for all feature types.

        Returns
        -------
        Dict[str, dict]
            Dictionary mapping feature keys to their result dictionaries

        Examples
        --------
        >>> selector_data = FeatureSelectorData("analysis")
        >>> all_results = selector_data.get_all_results()
        >>> for feature_key, results in all_results.items():
        ...     print(f"{feature_key}: {len(results['indices'])} selected")
        """
        return self.selection_results

    def has_results(self, feature_key: str = None) -> bool:
        """
        Check if selection results are available.

        Parameters
        ----------
        feature_key : str, optional
            Feature type key to check. If None, checks if any results exist.

        Returns
        -------
        bool
            True if results are available, False otherwise

        Examples
        --------
        >>> selector_data = FeatureSelectorData("analysis")
        >>> print(selector_data.has_results())  # Any results?
        False
        >>> print(selector_data.has_results("distances"))  # Specific feature?
        False
        """
        if feature_key is None:
            return len(self.selection_results) > 0
        return feature_key in self.selection_results

    def clear_results(self, feature_key: str = None) -> None:
        """
        Clear stored selection results.

        Parameters
        ----------
        feature_key : str, optional
            Feature type key to clear results for.
            If None, clears all results.

        Returns
        -------
        None
            Clears specified or all selection results

        Examples
        --------
        >>> selector_data = FeatureSelectorData("analysis")
        >>> selector_data.clear_results("distances")  # Clear only distances
        >>> selector_data.clear_results()  # Clear all results
        """
        if feature_key is None:
            self.selection_results.clear()
        elif feature_key in self.selection_results:
            del self.selection_results[feature_key]

    def __repr__(self) -> str:
        """
        String representation of FeatureSelectorData.

        Returns
        -------
        str
            String representation showing name and selection count

        Examples
        --------
        >>> selector_data = FeatureSelectorData("analysis")
        >>> selector_data.add_selection("distances", "res ALA")
        >>> print(repr(selector_data))
        FeatureSelectorData(name='analysis', features=1, selections=1)
        """
        summary = self.get_summary()
        results_count = len(self.selection_results)
        return (
            f"FeatureSelectorData(name='{self.name}', "
            f"features={summary['feature_count']}, "
            f"selections={summary['total_selections']}, "
            f"results={results_count})"
        )

    def set_reference_trajectory(self, reference_traj: int) -> None:
        """
        Set reference trajectory for metadata extraction.

        Parameters
        ----------
        reference_traj : int
            Trajectory index to use as reference for metadata

        Returns
        -------
        None
            Sets reference trajectory in the selector data
        """
        self.reference_trajectory = reference_traj

    def get_reference_trajectory(self) -> Optional[int]:
        """
        Get reference trajectory for metadata extraction.

        Returns
        -------
        Optional[int]
            Reference trajectory index, or None if not set
        """
        return self.reference_trajectory
    
    def set_n_columns(self, n_columns: int) -> None:
        """
        Set total number of columns in the final selection matrix.
        
        Parameters
        ----------
        n_columns : int
            Total number of columns across all features and trajectories
            
        Returns
        -------
        None
            Sets the n_columns attribute
        """
        self.n_columns = n_columns
    
    def get_n_columns(self) -> Optional[int]:
        """
        Get total number of columns in the final selection matrix.
        
        Returns
        -------
        Optional[int]
            Total number of columns, or None if not calculated yet
        """
        return self.n_columns

    def save(self, save_path: str) -> None:
        """
        Save FeatureSelectorData object to disk.

        Parameters
        ----------
        save_path : str
            Path where to save the FeatureSelectorData object

        Returns
        -------
        None
            Saves the FeatureSelectorData object to the specified path

        Examples
        --------
        >>> feature_selector_data.save('analysis_results/feature_selection.pkl')
        """
        DataUtils.save_object(self, save_path)

    def load(self, load_path: str) -> None:
        """
        Load FeatureSelectorData object from disk.

        Parameters
        ----------
        load_path : str
            Path to the saved FeatureSelectorData file

        Returns
        -------
        None
            Loads the FeatureSelectorData object from the specified path

        Examples
        --------
        >>> feature_selector_data.load('analysis_results/feature_selection.pkl')
        """
        DataUtils.load_object(self, load_path)

    def print_info(self) -> None:
        """
        Print comprehensive feature selector information.

        Parameters
        ----------
        None

        Returns
        -------
        None
            Prints feature selector information to console

        Examples
        --------
        >>> feature_selector_data.print_info()
        === FeatureSelectorData ===
        Name: analysis
        Feature Types: 2 (distances, contacts)
        Total Selections: 3
        Selection Results: Available for 2 feature types
        """
        if self._has_no_selections():
            print("No feature selections configured.")
            return

        self._print_selector_header()
        self._print_selector_details()
        self._print_results_info()

    def _has_no_selections(self) -> bool:
        """
        Check if no feature selections are configured.

        Returns
        -------
        bool
            True if no selections are available, False otherwise
        """
        return len(self.get_feature_keys()) == 0

    def _print_selector_header(self) -> None:
        """
        Print header with selector name.

        Returns
        -------
        None
        """
        print("=== FeatureSelectorData ===")
        print(f"Name: {self.name}")

    def _print_selector_details(self) -> None:
        """
        Print detailed selection information.

        Returns
        -------
        None
        """
        summary = self.get_summary()
        feature_keys = self.get_feature_keys()
        
        print(f"Feature Types: {summary['feature_count']} ({', '.join(feature_keys)})")
        print(f"Total Selections: {summary['total_selections']}")
        
        if summary['reduced_selections'] > 0:
            print(f"  - Using reduced data: {summary['reduced_selections']}")
            print(f"  - Using original data: {summary['original_selections']}")

        # Show reference trajectory if set
        if hasattr(self, 'reference_trajectory') and self.reference_trajectory is not None:
            print(f"Reference Trajectory: {self.reference_trajectory}")

        # Show column count if available
        if hasattr(self, 'n_columns') and self.n_columns is not None:
            print(f"Matrix Columns: {self.n_columns}")

    def _print_results_info(self) -> None:
        """
        Print information about stored results.

        Returns
        -------
        None
        """
        if len(self.selection_results) > 0:
            result_features = list(self.selection_results.keys())
            print(f"Selection Results: Available for {len(result_features)} feature types ({', '.join(result_features)})")
        else:
            print("Selection Results: Not computed yet")
