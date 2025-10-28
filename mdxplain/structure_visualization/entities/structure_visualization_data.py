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
Structure visualization data entity for storing PDB paths.

This module contains the StructureVisualizationData class that stores
PDB file paths generated for structure visualization. Each instance
represents a named visualization session with PDBs for different
sub-comparisons.
"""

from typing import Dict, Optional


class StructureVisualizationData:
    """
    Data entity for storing structure visualization PDB paths.

    Stores paths to generated PDB files for structure visualization,
    organized by sub-comparison identifiers. Each StructureVisualizationData
    represents a single visualization session with potentially multiple
    structures.

    Attributes
    ----------
    name : str
        Name identifier for this visualization session
    feature_importance_name : str
        Name of feature importance analysis used for beta-factors
    pdb_paths : Dict[str, str]
        Mapping from sub-comparison identifier to PDB file path

    Examples
    --------
    >>> viz_data = StructureVisualizationData("my_viz", "dt_analysis")
    >>> viz_data.add_pdb("cluster_0_vs_rest", "/path/to/structure.pdb")
    >>> path = viz_data.get_pdb("cluster_0_vs_rest")
    >>> all_paths = viz_data.get_all_pdbs()
    """

    def __init__(self, name: str, feature_importance_name: str):
        """
        Initialize structure visualization data with given name.

        Parameters
        ----------
        name : str
            Name identifier for this visualization session
        feature_importance_name : str
            Name of feature importance analysis used for beta-factors

        Returns
        -------
        None
            Initializes StructureVisualizationData with empty PDB paths

        Examples
        --------
        >>> viz_data = StructureVisualizationData("my_viz", "dt_analysis")
        >>> print(viz_data.name)
        'my_viz'
        >>> print(viz_data.feature_importance_name)
        'dt_analysis'
        """
        self.name = name
        self.feature_importance_name = feature_importance_name
        self.pdb_paths: Dict[str, str] = {}

    def add_pdb(self, sub_comparison: str, pdb_path: str) -> None:
        """
        Add PDB path for a sub-comparison.

        Parameters
        ----------
        sub_comparison : str
            Sub-comparison identifier
        pdb_path : str
            Path to PDB file

        Returns
        -------
        None
            Adds PDB path to internal storage

        Examples
        --------
        >>> viz_data = StructureVisualizationData("my_viz")
        >>> viz_data.add_pdb("cluster_0_vs_rest", "/path/to/structure.pdb")
        """
        self.pdb_paths[sub_comparison] = pdb_path

    def get_pdb(self, sub_comparison: str) -> Optional[str]:
        """
        Get PDB path for a sub-comparison.

        Parameters
        ----------
        sub_comparison : str
            Sub-comparison identifier

        Returns
        -------
        Optional[str]
            PDB file path, or None if not found

        Examples
        --------
        >>> viz_data = StructureVisualizationData("my_viz")
        >>> viz_data.add_pdb("cluster_0_vs_rest", "/path/to/structure.pdb")
        >>> path = viz_data.get_pdb("cluster_0_vs_rest")
        >>> print(path)
        '/path/to/structure.pdb'
        """
        return self.pdb_paths.get(sub_comparison)

    def get_all_pdbs(self) -> Dict[str, str]:
        """
        Get all PDB paths.

        Returns
        -------
        Dict[str, str]
            Dictionary mapping sub-comparison to PDB path

        Examples
        --------
        >>> viz_data = StructureVisualizationData("my_viz")
        >>> viz_data.add_pdb("cluster_0", "/path/to/c0.pdb")
        >>> viz_data.add_pdb("cluster_1", "/path/to/c1.pdb")
        >>> all_paths = viz_data.get_all_pdbs()
        >>> print(len(all_paths))
        2
        """
        return self.pdb_paths.copy()

    def has_pdb(self, sub_comparison: str) -> bool:
        """
        Check if PDB exists for sub-comparison.

        Parameters
        ----------
        sub_comparison : str
            Sub-comparison identifier

        Returns
        -------
        bool
            True if PDB path exists, False otherwise

        Examples
        --------
        >>> viz_data = StructureVisualizationData("my_viz")
        >>> viz_data.add_pdb("cluster_0", "/path/to/c0.pdb")
        >>> viz_data.has_pdb("cluster_0")
        True
        >>> viz_data.has_pdb("cluster_1")
        False
        """
        return sub_comparison in self.pdb_paths
