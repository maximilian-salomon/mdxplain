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
structures (from FI sub-comparisons or data selectors).
"""

from typing import Dict, Optional, List, Any


class StructureVisualizationData:
    """
    Data entity for storing structure visualization PDB paths.

    Stores paths to generated PDB files for structure visualization,
    organized by structure identifiers. Supports two source types:
    - "feature_importance": From FI analysis sub-comparisons
    - "feature": From feature/data selectors

    Attributes
    ----------
    name : str
        Name identifier for this visualization session
    source_type : str
        Source type: "feature_importance" or "feature"
    feature_importance_name : str | None
        Name of FI analysis (for FI source type)
    selector_centroid : str | None
        Feature selector for centroids (for feature source type)
    selector_features : str | None
        Feature selector for highlights (for feature source type)
    pdb_paths : Dict[str, str]
        Mapping from structure identifier to PDB file path
    feature_info : List[Dict[str, Any]]
        List of feature dicts with metadata for visualization

    Examples
    --------
    >>> # Feature importance mode
    >>> viz_data = StructureVisualizationData("my_viz", "dt_analysis")
    >>> viz_data.add_pdb("cluster_0_vs_rest", "/path/to/structure.pdb")

    >>> # Feature mode
    >>> viz_data = StructureVisualizationData.from_selectors(
    ...     "my_viz", "coords_all", "distances"
    ... )
    >>> viz_data.add_pdb("cluster_0", "/path/to/c0.pdb")
    """

    def __init__(
        self,
        name: str,
        feature_importance_name: str = None,
        selector_centroid: str = None,
        selector_features: str = None
    ):
        """
        Initialize structure visualization data.

        Use either feature_importance_name (FI mode) OR selector_centroid
        (feature mode), not both.

        Parameters
        ----------
        name : str
            Name identifier for this visualization session
        feature_importance_name : str, optional
            Name of FI analysis (for FI source type)
        selector_centroid : str, optional
            Feature selector for centroids (for feature source type)
        selector_features : str, optional
            Feature selector for highlights (for feature source type)

        Returns
        -------
        None
            Initializes StructureVisualizationData with empty PDB paths

        Examples
        --------
        >>> # Feature importance mode
        >>> viz_data = StructureVisualizationData("my_viz", "dt_analysis")

        >>> # Feature mode with features
        >>> viz_data = StructureVisualizationData(
        ...     "my_viz",
        ...     selector_centroid="coords_all",
        ...     selector_features="distances"
        ... )

        >>> # Feature mode without features
        >>> viz_data = StructureVisualizationData(
        ...     "my_viz",
        ...     selector_centroid="coords_all"
        ... )
        """
        self.name = name

        # Determine source type
        if feature_importance_name is not None:
            self.source_type = "feature_importance"
            self.feature_importance_name = feature_importance_name
            self.selector_centroid = None
            self.selector_features = None
        else:
            self.source_type = "feature"
            self.feature_importance_name = None
            self.selector_centroid = selector_centroid
            self.selector_features = selector_features

        self.pdb_paths: Dict[str, str] = {}
        self.feature_info: List[Dict[str, Any]] = []

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

    def add_feature_info(self, features: List[Dict[str, Any]]) -> None:
        """
        Add feature information for visualization.

        Parameters
        ----------
        features : List[Dict[str, Any]]
            List of feature dictionaries with metadata

        Returns
        -------
        None
            Stores feature info for later visualization

        Examples
        --------
        >>> viz_data = StructureVisualizationData(
        ...     "my_viz", selector_centroid="coords_all"
        ... )
        >>> features = [
        ...     {"feature_name": "ALA_5_CA-GLU_10_CA", "residue_seqids": [5, 10]},
        ...     {"feature_name": "GLY_3_phi", "residue_seqids": [3]}
        ... ]
        >>> viz_data.add_feature_info(features)
        """
        self.feature_info = features

    def get_feature_info(self) -> List[Dict[str, Any]]:
        """
        Get stored feature information.

        Returns
        -------
        List[Dict[str, Any]]
            List of feature dictionaries with metadata

        Examples
        --------
        >>> viz_data = StructureVisualizationData(
        ...     "my_viz", selector_centroid="coords_all"
        ... )
        >>> viz_data.add_feature_info([{"feature_name": "test"}])
        >>> features = viz_data.get_feature_info()
        >>> len(features)
        1
        """
        return self.feature_info
