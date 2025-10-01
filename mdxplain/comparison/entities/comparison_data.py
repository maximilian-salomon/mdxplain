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
Comparison data entity for storing comparison configurations.

This module contains the ComparisonData class that stores comparison
configurations including sub-comparisons, feature selectors, and
data selectors for ML analysis.
"""

from typing import Dict, List, Any, Optional, Tuple

from ...utils.data_utils import DataUtils

class ComparisonData:
    """
    Data entity for storing comparison configurations for ML analysis.

    Stores comparison configurations that define how to create ML-ready
    datasets from different data selections. A single ComparisonData
    can contain multiple sub-comparisons (e.g., in one-vs-rest mode).

    Attributes
    ----------
    name : str
        Name identifier for this comparison
    mode : str
        Comparison mode: "binary", "pairwise", "one_vs_rest", "multiclass"
    feature_selector : str
        Name of the feature selector to use for columns
    data_selectors : List[str]
        Names of data selectors involved in the comparison
    sub_comparisons : List[Dict[str, Any]]
        List of sub-comparisons with their configurations

    Examples
    --------
    Binary comparison:
    >>> comp_data = ComparisonData("folded_vs_unfolded", "binary", "key_features")
    >>> comp_data.add_sub_comparison("folded_vs_unfolded", ["folded"], ["unfolded"])

    One-vs-rest comparison:
    >>> comp_data = ComparisonData("conformations", "one_vs_rest", "all_features")
    >>> # This will contain multiple sub-comparisons automatically
    """

    def __init__(
        self,
        name: str,
        mode: str,
        feature_selector: str,
        data_selectors: Optional[List[str]] = None,
    ):
        """
        Initialize comparison data with basic configuration.

        ComparisonData is a pure metadata container that stores configuration
        for ML comparisons without processing actual data. Data processing
        is handled by ComparisonManager.

        Parameters
        ----------
        name : str
            Name identifier for this comparison
        mode : str
            Comparison mode: "binary", "pairwise", "one_vs_rest", "multiclass"
        feature_selector : str
            Name of the feature selector to use for columns
        data_selectors : List[str], optional
            Names of data selectors involved in the comparison

        Returns
        -------
        None
            Initializes ComparisonData with given configuration

        Examples
        --------
        >>> comp_data = ComparisonData(
        ...     "systems_comparison", "pairwise", "important_features",
        ...     ["system_A", "system_B", "system_C"]
        ... )
        """
        self.name = name
        self.mode = mode
        self.feature_selector = feature_selector
        self.data_selectors = data_selectors or []
        self.sub_comparisons: List[Dict[str, Any]] = []

        # Validate mode
        # TODO: We could use an Enum for modes
        valid_modes = ["binary", "pairwise", "one_vs_rest", "multiclass"]
        if mode not in valid_modes:
            raise ValueError(f"Invalid mode '{mode}'. Valid modes: {valid_modes}")

    def add_sub_comparison(
        self,
        sub_name: str,
        group1_selectors: List[str],
        group2_selectors: List[str],
        labels: Optional[Tuple[int, int]] = None,
    ) -> None:
        """
        Add a sub-comparison to this comparison.

        Parameters
        ----------
        sub_name : str
            Name of the sub-comparison
        group1_selectors : List[str]
            Data selector names for group 1
        group2_selectors : List[str]
            Data selector names for group 2
        labels : Tuple[int, int], optional
            Label values for (group1, group2). Defaults to (0, 1)

        Returns
        -------
        None
            Adds sub-comparison to the list

        Examples
        --------
        >>> comp_data.add_sub_comparison(
        ...     "folded_vs_unfolded", ["folded"], ["unfolded"], (0, 1)
        ... )
        """
        if labels is None:
            labels = (0, 1)

        sub_comp = {
            "name": sub_name,
            "group1_selectors": group1_selectors,
            "group2_selectors": group2_selectors,
            "labels": labels,
        }

        self.sub_comparisons.append(sub_comp)

    def get_sub_comparison(self, sub_name: str) -> Optional[Dict[str, Any]]:
        """
        Get a specific sub-comparison by name.

        Parameters
        ----------
        sub_name : str
            Name of the sub-comparison to retrieve

        Returns
        -------
        Dict[str, Any] or None
            Sub-comparison dictionary or None if not found

        Examples
        --------
        >>> sub_comp = comp_data.get_sub_comparison("folded_vs_rest")
        >>> if sub_comp:
        ...     print(f"Group 1: {sub_comp['group1_selectors']}")
        """
        for sub_comp in self.sub_comparisons:
            if sub_comp["name"] == sub_name:
                return sub_comp
        return None

    def list_sub_comparisons(self) -> List[str]:
        """
        List names of all sub-comparisons.

        Returns
        -------
        List[str]
            List of sub-comparison names

        Examples
        --------
        >>> names = comp_data.list_sub_comparisons()
        >>> print(f"Available sub-comparisons: {names}")
        """
        return [sub_comp["name"] for sub_comp in self.sub_comparisons]

    def get_comparison_info(self) -> Dict[str, Any]:
        """
        Get summary information about this comparison.

        Returns
        -------
        Dict[str, Any]
            Dictionary with comparison summary information

        Examples
        --------
        >>> info = comp_data.get_comparison_info()
        >>> print(f"Mode: {info['mode']}")
        >>> print(f"Sub-comparisons: {info['n_sub_comparisons']}")
        """
        return {
            "name": self.name,
            "mode": self.mode,
            "feature_selector": self.feature_selector,
            "data_selectors": self.data_selectors,
            "n_sub_comparisons": len(self.sub_comparisons),
            "sub_comparison_names": self.list_sub_comparisons(),
        }

    def __len__(self) -> int:
        """
        Get number of sub-comparisons.

        Returns
        -------
        int
            Number of sub-comparisons

        Examples
        --------
        >>> print(f"Number of sub-comparisons: {len(comp_data)}")
        """
        return len(self.sub_comparisons)

    def __contains__(self, sub_name: str) -> bool:
        """
        Check if a sub-comparison exists.

        Parameters
        ----------
        sub_name : str
            Name of the sub-comparison to check

        Returns
        -------
        bool
            True if sub-comparison exists, False otherwise

        Examples
        --------
        >>> if "folded_vs_rest" in comp_data:
        ...     print("Sub-comparison exists")
        """
        return sub_name in self.list_sub_comparisons()

    def __repr__(self) -> str:
        """
        String representation of the ComparisonData.

        Returns
        -------
        str
            String representation

        Examples
        --------
        >>> print(repr(comp_data))
        ComparisonData(name='folded_analysis', mode='one_vs_rest', n_sub=4)
        """
        return (
            f"ComparisonData(name='{self.name}', mode='{self.mode}', "
            f"n_sub={len(self.sub_comparisons)})"
        )

    def save(self, save_path: str) -> None:
        """
        Save ComparisonData object to disk.

        Parameters
        ----------
        save_path : str
            Path where to save the ComparisonData object

        Returns
        -------
        None
            Saves the ComparisonData object to the specified path

        Examples
        --------
        >>> comparison_data.save('analysis_results/folded_analysis.pkl')
        """
        DataUtils.save_object(self, save_path)

    def load(self, load_path: str) -> None:
        """
        Load ComparisonData object from disk.

        Parameters
        ----------
        load_path : str
            Path to the saved ComparisonData file

        Returns
        -------
        None
            Loads the ComparisonData object from the specified path

        Examples
        --------
        >>> comparison_data.load('analysis_results/folded_analysis.pkl')
        """
        DataUtils.load_object(self, load_path)

    def print_info(self) -> None:
        """
        Print comprehensive comparison information.

        Parameters
        ----------
        None

        Returns
        -------
        None
            Prints comparison information to console

        Examples
        --------
        >>> comparison_data.print_info()
        === ComparisonData ===
        Name: folded_analysis
        Comparison Mode: one_vs_rest
        Feature Selector: key_features
        Data Selectors: 3 (folded, intermediate, unfolded)
        Sub-Comparisons: 3 (folded_vs_rest, intermediate_vs_rest, unfolded_vs_rest)
        """
        if self._has_no_comparisons():
            print("No comparison data available.")
            return

        self._print_comparison_header()
        self._print_comparison_details()
        self._print_sub_comparison_details()

    def _has_no_comparisons(self) -> bool:
        """
        Check if no comparisons are configured.

        Returns
        -------
        bool
            True if no comparisons are available, False otherwise
        """
        return len(self.sub_comparisons) == 0

    def _print_comparison_header(self) -> None:
        """
        Print header with comparison name.

        Returns
        -------
        None
        """
        print("=== ComparisonData ===")
        print(f"Name: {self.name}")

    def _print_comparison_details(self) -> None:
        """
        Print detailed comparison configuration.

        Returns
        -------
        None
        """
        print(f"Comparison Mode: {self.mode}")
        print(f"Feature Selector: {self.feature_selector}")
        
        if self.data_selectors:
            data_selector_names = ", ".join(self.data_selectors)
            print(f"Data Selectors: {len(self.data_selectors)} ({data_selector_names})")
        else:
            print("Data Selectors: None configured")

    def _print_sub_comparison_details(self) -> None:
        """
        Print information about sub-comparisons.

        Returns
        -------
        None
        """
        sub_names = self.list_sub_comparisons()
        if sub_names:
            sub_names_str = ", ".join(sub_names)
            print(f"Sub-Comparisons: {len(sub_names)} ({sub_names_str})")
        else:
            print("Sub-Comparisons: None configured")
