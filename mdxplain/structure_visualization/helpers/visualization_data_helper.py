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
Visualization data helper for structure visualization operations.

This module provides helper utilities for preparing visualization data,
including PDB info preparation, feature color assignment, and feature
extraction from selectors. Used by both StructureVizFeatureImportanceService
and StructureVizFeatureService.
"""

from __future__ import annotations

import os
from typing import Dict, List, Any, TYPE_CHECKING

if TYPE_CHECKING:
    from ...pipeline.entities.pipeline_data import PipelineData
    from ..entities.structure_visualization_data import StructureVisualizationData

from ...utils.color_utils import ColorUtils
from ...utils.feature_metadata_utils import FeatureMetadataUtils
from ...utils.top_features_utils import TopFeaturesUtils


class VisualizationDataHelper:
    """
    Helper class for structure visualization data operations.

    Provides static methods for preparing PDB info dictionaries with
    colors, assigning feature colors, and extracting features from
    selectors. Used by both Feature Importance and Feature-based services.

    Examples
    --------
    >>> pdb_info = VisualizationDataHelper.prepare_pdb_info_from_viz_data(
    ...     viz_data
    ... )
    >>> colors = VisualizationDataHelper.assign_feature_colors(features, 5)
    """

    @staticmethod
    def prepare_pdb_info_from_comp_data(
        viz_data: StructureVisualizationData,
        comp_data: Any
    ) -> Dict[str, Dict[str, str]]:
        """
        Prepare PDB info from visualization and comparison data.

        Creates dictionary mapping sub-comparison names to absolute PDB
        paths and assigned colors. Used by Feature Importance Service.

        Parameters
        ----------
        viz_data : StructureVisualizationData
            Visualization data containing PDB paths
        comp_data : ComparisonData
            Comparison data with sub-comparisons

        Returns
        -------
        Dict[str, Dict[str, str]]
            Dictionary with structure info:
            - Keys: sub-comparison identifiers
            - Values: {"path": absolute_pdb_path, "color": hex_color}

        Examples
        --------
        >>> pdb_info = VisualizationDataHelper.prepare_pdb_info_from_comp_data(
        ...     viz_data, comp_data
        ... )
        >>> pdb_info["cluster_0_vs_rest"]
        {'path': '/abs/path/to/c0.pdb', 'color': '#bf4242'}

        Notes
        -----
        - Paths converted to absolute using os.path.abspath()
        - Colors generated using ColorUtils.generate_distinct_colors()
        - Color assignment follows sub-comparison order
        """
        pdb_info = {}

        # Generate distinct colors for structures
        n_structures = len(comp_data.sub_comparisons)
        structure_colors = ColorUtils.generate_distinct_colors(n_structures)

        # Build pdb_info with assigned colors
        for i, sub_comp in enumerate(comp_data.sub_comparisons):
            comp_id = sub_comp["name"]
            pdb_path = viz_data.get_pdb(comp_id)

            if pdb_path is not None:
                pdb_info[comp_id] = {
                    "path": os.path.abspath(pdb_path),
                    "color": structure_colors[i]
                }

        return pdb_info

    @staticmethod
    def prepare_pdb_info_from_viz_data(
        viz_data: StructureVisualizationData
    ) -> Dict[str, Dict[str, str]]:
        """
        Prepare PDB info from visualization data only.

        Creates dictionary with structure info from viz_data.get_all_pdbs().
        Used by Feature Service where no comparison data is available.

        Parameters
        ----------
        viz_data : StructureVisualizationData
            Visualization data with PDB paths

        Returns
        -------
        Dict[str, Dict[str, str]]
            Dictionary with structure info:
            - Keys: structure identifiers
            - Values: {"path": absolute_pdb_path, "color": hex_color}

        Examples
        --------
        >>> viz_data.add_pdb("cluster_0", "/path/to/c0.pdb")
        >>> viz_data.add_pdb("cluster_1", "/path/to/c1.pdb")
        >>> pdb_info = VisualizationDataHelper.prepare_pdb_info_from_viz_data(
        ...     viz_data
        ... )
        >>> len(pdb_info)
        2

        Notes
        -----
        - Paths converted to absolute using os.path.abspath()
        - Colors generated using ColorUtils.generate_distinct_colors()
        - Color assignment follows dictionary iteration order
        """
        pdb_info = {}

        # Get all PDB paths
        all_pdbs = viz_data.get_all_pdbs()

        # Generate distinct colors for structures
        structure_colors = ColorUtils.generate_distinct_colors(len(all_pdbs))

        # Build pdb_info with assigned colors
        for i, (struct_id, pdb_path) in enumerate(all_pdbs.items()):
            pdb_info[struct_id] = {
                "path": os.path.abspath(pdb_path),
                "color": structure_colors[i]
            }

        return pdb_info

    @staticmethod
    def assign_feature_colors(
        top_features: List[Dict[str, Any]],
        n_features: int,
        offset: int = 0
    ) -> Dict[str, str]:
        """
        Assign distinct colors to top features.

        Creates color mapping for features using visually distinct colors
        for highlighting in visualizations. Uses ColorUtils to generate
        perceptually distinct colors.

        Parameters
        ----------
        top_features : List[Dict[str, Any]]
            List of top feature dictionaries with 'feature_name' key
        n_features : int
            Number of features to assign colors to
        offset : int, default=0
            Color offset for continuous coloring across feature groups.
            Used to assign different colors to global vs local features.

        Returns
        -------
        Dict[str, str]
            Mapping from feature name to HEX color string

        Examples
        --------
        >>> # Global features get colors 0-2
        >>> global_features = [{"feature_name": "ALA_5_CA-GLU_10_CA"}]
        >>> global_colors = VisualizationDataHelper.assign_feature_colors(
        ...     global_features, 3, offset=0
        ... )
        >>> # Local features get colors 3-5
        >>> local_features = [{"feature_name": "GLY_3_phi"}]
        >>> local_colors = VisualizationDataHelper.assign_feature_colors(
        ...     local_features, 3, offset=3
        ... )

        Notes
        -----
        - Uses ColorUtils.generate_distinct_colors() for color generation
        - If feature_name missing, uses "feature_{i}" as fallback
        - Only first n_features are assigned colors
        - Offset enables continuous coloring: global (0-N), local (N-M)
        """
        colors_dict = ColorUtils.generate_distinct_colors(n_features + offset)

        feature_colors = {}
        for i, feature in enumerate(top_features[:n_features]):
            feature_name = feature.get("feature_name", f"feature_{i}")
            feature_colors[feature_name] = colors_dict[i + offset]

        return feature_colors

    @staticmethod
    def extract_local_features_and_colors(
        pipeline_data: PipelineData,
        fi_data: Any,
        comp_data: Any,
        n_top_local: int,
        n_top_global: int,
        feature_own_color: bool = True  # Kept for API compatibility
    ) -> tuple:
        """
        Extract local top features and colors per cluster.

        Extracts cluster-specific top features from feature importance data
        and assigns colors with optional offset for continuous coloring.

        Parameters
        ----------
        pipeline_data : PipelineData
            Pipeline data object
        fi_data : FeatureImportanceData
            Feature importance data
        comp_data : ComparisonData
            Comparison data with sub-comparisons
        n_top_local : int
            Number of local features per cluster
        n_top_global : int
            Number of global features (for color offset)
        feature_own_color : bool
            Not used for color assignment (kept for API compatibility).
            Colors always use offset to avoid conflicts with global features.

        Returns
        -------
        tuple
            (top_features_local, feature_colors_local) where:
            - top_features_local: Dict[str, List[Dict]] - features per cluster
            - feature_colors_local: Dict[str, Dict[str, str]] - colors per cluster

        Examples
        --------
        >>> local_feats, local_colors = VisualizationDataHelper.extract_local_features_and_colors(
        ...     pipeline_data, fi_data, comp_data, n_top_local=3,
        ...     n_top_global=3, feature_own_color=False
        ... )
        >>> local_feats["cluster_0"]
        [{'feature_name': 'ALA_5-GLU_10', ...}]

        Notes
        -----
        Returns empty dicts if n_top_local is 0.
        Color offset enables continuous coloring: global (0-N), local (N-M).
        """
        top_features_local = {}
        feature_colors_local = {}

        if n_top_local > 0:
            for sub_comp in comp_data.sub_comparisons:
                comp_id = sub_comp["name"]

                local_features = TopFeaturesUtils.get_top_features_with_names(
                    pipeline_data, fi_data, comp_id, n_top_local
                )
                top_features_local[comp_id] = local_features

                # Always use offset to avoid color conflicts with global features
                feature_colors_local[comp_id] = VisualizationDataHelper.assign_feature_colors(
                    local_features, n_top_local, offset=n_top_global
                )

        return top_features_local, feature_colors_local

    @staticmethod
    def extract_features_from_selector(
        pipeline_data: PipelineData,
        selector_name: str
    ) -> List[Dict[str, Any]]:
        """
        Extract all features from feature selector.

        Uses PUBLIC API get_selected_metadata() to retrieve already
        selected and formatted feature metadata, same approach as
        Feature Importance Service.

        Parameters
        ----------
        pipeline_data : PipelineData
            Pipeline data object
        selector_name : str
            Feature selector name

        Returns
        -------
        List[Dict[str, Any]]
            List of feature dicts with metadata:
            - feature_index: int
            - feature_name: str
            - feature_type: str
            - residue_seqids: List[int]
            - residue_indices: List[int]

        Raises
        ------
        KeyError
            If selector not found in selected_feature_data

        Examples
        --------
        >>> features = VisualizationDataHelper.extract_features_from_selector(
        ...     pipeline_data, "important_distances"
        ... )
        >>> features[0]["feature_name"]
        'ALA_5_CA-GLU_10_CA'

        Notes
        -----
        This method uses the same PUBLIC API as Feature Importance Service:
        pipeline_data.get_selected_metadata(). This ensures consistency
        and avoids index translation issues.
        """
        # Validate selector exists
        if selector_name not in pipeline_data.selected_feature_data:
            raise KeyError(
                f"Feature selector '{selector_name}' not found"
            )

        # Get selected metadata (PUBLIC API - same as FI Service)
        metadata = pipeline_data.get_selected_metadata(selector_name)

        # Extract all features from metadata
        features = []
        for idx in range(len(metadata)):
            feature_info = {
                "feature_index": idx,
                "feature_name": FeatureMetadataUtils.get_feature_name(
                    metadata, idx
                ),
                "feature_type": FeatureMetadataUtils.get_feature_type(
                    metadata, idx
                ),
            }

            # Get residue information
            residues = FeatureMetadataUtils.get_feature_residues(
                metadata, idx
            )
            feature_info["residue_seqids"] = [res["seqid"] for res in residues]
            feature_info["residue_indices"] = [res["index"] for res in residues]

            features.append(feature_info)

        return features
