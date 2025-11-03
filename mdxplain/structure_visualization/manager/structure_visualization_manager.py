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
Structure visualization manager for molecular structure analysis.

This manager coordinates structure visualization workflows through
specialized services for feature importance and feature-based approaches.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ...pipeline.entities.pipeline_data import PipelineData

from ..helper.residue_importance_calculator import ResidueImportanceCalculator
from ..helper.pdb_beta_factor_helper import PdbBetaFactorHelper
from ...utils.top_features_utils import TopFeaturesUtils


class StructureVisualizationManager:
    """
    Manager for structure visualization workflows.

    Provides two specialized services:
    
    - feature_importance: Visualizations based on feature importance analysis
    - feature: Visualizations based on feature and data selectors

    Warning
    -------
    When using PipelineManager, the pipeline_data parameter is
    automatically injected. Do NOT provide it manually.

    Examples
    --------
    >>> # Feature importance approach
    >>> pipeline.structure_visualization.feature_importance.create_pdb_with_beta_factors(
    ...     "my_viz", "dt_analysis", n_top=10
    ... )
    >>> pipeline.structure_visualization.feature_importance.visualize_nglview_jupyter(
    ...     "my_viz", n_top_global=3
    ... )

    >>> # Feature selector approach
    >>> pipeline.structure_visualization.feature.create_representative_pdbs(
    ...     "my_viz",
    ...     data_selectors=["cluster_0", "cluster_1"],
    ...     selector_centroid="coords_all",
    ...     selector_features="distances"
    ... )
    >>> pipeline.structure_visualization.feature.visualize_nglview_jupyter("my_viz")
    """

    def __init__(
        self,
        use_memmap: bool = False,
        chunk_size: int = 2000,
        cache_dir: str = "./cache"
    ):
        """
        Initialize structure visualization manager.

        Creates manager instance and sets up dedicated output directory
        for structure visualization files (PDBs, scripts, etc.).

        Parameters
        ----------
        use_memmap : bool, default=False
            Whether to use memory mapping for large datasets
        chunk_size : int, default=2000
            Chunk size for processing large arrays
        cache_dir : str, default="./cache"
            Base cache directory. Manager creates subdirectory
            "structure_viz" as dedicated output directory.

        Returns
        -------
        None
            Initializes manager instance with output directory

        Notes
        -----
        Output directory is set to cache_dir/structure_viz and created
        automatically during initialization.
        """
        self.use_memmap = use_memmap
        self.chunk_size = chunk_size
        self.cache_dir = cache_dir

        # Create dedicated output directory for structure visualization
        self.output_dir = os.path.join(cache_dir, "structure_viz")
        os.makedirs(self.output_dir, exist_ok=True)

    @property
    def feature_importance(self):
        """
        Service for feature-importance-based visualization.

        Provides methods for creating and visualizing molecular structures
        with beta-factors derived from feature importance analysis.

        Returns
        -------
        StructureVizFeatureImportanceService
            Service instance for FI-based visualization

        Examples
        --------
        >>> pipeline.structure_visualization.feature_importance.create_pdb_with_beta_factors(
        ...     "my_viz", "dt_analysis", n_top=10
        ... )
        """
        from ..services.structure_viz_feature_importance_service import (
            StructureVizFeatureImportanceService,
        )
        return StructureVizFeatureImportanceService(self, None)

    @property
    def feature(self):
        """
        Service for feature-selector-based visualization.

        Provides methods for creating and visualizing molecular structures
        based on feature selectors and data selectors without requiring
        feature importance analysis.

        Returns
        -------
        StructureVizFeatureService
            Service instance for feature-based visualization

        Examples
        --------
        >>> pipeline.structure_visualization.feature.create_representative_pdbs(
        ...     "my_viz",
        ...     data_selectors=["cluster_0", "cluster_1"],
        ...     selector_centroid="coords_all",
        ...     selector_features="distances"
        ... )
        """
        from ..services.structure_viz_feature_service import (
            StructureVizFeatureService,
        )
        return StructureVizFeatureService(self, None)

    def _calculate_beta_factors(
        self,
        pipeline_data: PipelineData,
        fi_data: Any,
        comparison_identifier: str,
        traj_idx: int,
        n_top: int
    ) -> Any:
        """
        Calculate per-atom beta-factors from feature importance.

        Computes beta-factor values for all atoms in a structure based
        on residue importance scores derived from top important features.

        Parameters
        ----------
        pipeline_data : PipelineData
            Pipeline data object
        fi_data : FeatureImportanceData
            Feature importance data
        comparison_identifier : str
            Sub-comparison identifier
        traj_idx : int
            Trajectory index for topology
        n_top : int
            Number of top features to consider

        Returns
        -------
        np.ndarray
            Per-atom beta-factors (0-100 scale)
        """
        top_features = TopFeaturesUtils.get_top_features_with_names(
            pipeline_data, fi_data, comparison_identifier, n_top
        )
        topology = PdbBetaFactorHelper.get_topology(pipeline_data, traj_idx)
        return ResidueImportanceCalculator.calculate_beta_factors_from_features(
            top_features, topology
        )
