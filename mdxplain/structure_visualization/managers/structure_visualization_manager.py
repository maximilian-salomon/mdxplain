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

This manager coordinates structure visualization workflows including
PDB creation with beta-factor coloring and interactive NGLView widgets
for Jupyter notebooks.
"""

from __future__ import annotations

import os
from typing import Dict, List, Tuple, Any, TYPE_CHECKING

if TYPE_CHECKING:
    from ...pipeline.entities.pipeline_data import PipelineData

from ..entities.structure_visualization_data import StructureVisualizationData
from ..helpers.residue_importance_calculator import ResidueImportanceCalculator
from ..helpers.pdb_beta_factor_helper import PdbBetaFactorHelper
from ..helpers.validation_helper import ValidationHelper
from ...utils.top_features_utils import TopFeaturesUtils
from ...utils.color_utils import ColorUtils
from ...utils.data_utils import DataUtils


class StructureVisualizationManager:
    """
    Manager for structure visualization workflows.

    Coordinates PDB creation with feature importance beta-factors and
    interactive 3D visualization in Jupyter notebooks using NGLView.

    Warning
    -------
    When using PipelineManager, the pipeline_data parameter is
    automatically injected. Do NOT provide it manually.

    Examples
    --------
    >>> # Via PipelineManager
    >>> pipeline.structure_visualization.create_pdb_with_beta_factors(
    ...     "my_viz", "dt_analysis", n_top=10
    ... )
    >>> pipeline.structure_visualization.visualize_nglview_jupyter(
    ...     "my_viz", n_top_global=3
    ... )
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

    def create_pdb_with_beta_factors(
        self,
        pipeline_data: PipelineData,
        structure_viz_name: str,
        feature_importance_name: str,
        n_top: int = 10,
        representative_mode: str = "best",
        output_dir: str = None
    ) -> None:
        """
        Create PDB files with beta-factor coloring for all comparisons.

        Generates PDB files with beta-factors encoding residue importance
        for all sub-comparisons in a feature importance analysis. Results
        are stored in pipeline_data for later visualization.

        Warning
        -------
        When using PipelineManager, do NOT provide the pipeline_data parameter.

        Parameters
        ----------
        pipeline_data : PipelineData
            Pipeline data object (auto-injected by PipelineManager)
        structure_viz_name : str
            Name for this visualization session
        feature_importance_name : str
            Name of feature importance analysis to visualize
        n_top : int, default=10
            Number of top features to consider for beta-factors
        representative_mode : str, default="best"
            Mode for selecting representative frames:
            - "best": Frame maximizing top feature values
            - "centroid": Frame closest to cluster centroid
        output_dir : str, optional
            Output directory for PDB files (defaults to cache_dir/structure_viz)

        Returns
        -------
        None
            Stores PDB paths in pipeline_data.structure_visualization_data

        Examples
        --------
        >>> pipeline.structure_visualization.create_pdb_with_beta_factors(
        ...     "my_viz", "dt_analysis", n_top=10
        ... )
        >>> # Access later for visualization
        >>> data = pipeline.data.structure_visualization_data["my_viz"]

        Notes
        -----
        - Creates one PDB per sub-comparison
        - Beta-factors encode residue importance (0-100 scale)
        - Representative frames selected based on representative_mode
        """
        # Use provided output_dir or default to manager's output_dir
        if output_dir is None:
            output_dir = self.output_dir

        fi_data = pipeline_data.feature_importance_data[feature_importance_name]
        comp_data = pipeline_data.comparison_data[fi_data.comparison_name]

        # Create visualization data entity
        viz_data = StructureVisualizationData(structure_viz_name, feature_importance_name)

        # Process each sub-comparison
        for sub_comp in comp_data.sub_comparisons:
            comp_id = sub_comp["name"]

            # Get representative frame
            traj_idx, frame_idx = fi_data.get_representative_frame(
                pipeline_data, comp_id, representative_mode,
                n_top, self.use_memmap, self.chunk_size
            )

            # Calculate beta-factors
            beta_factors = self._calculate_beta_factors(
                pipeline_data, fi_data, comp_id, traj_idx, n_top
            )

            # Create PDB
            pdb_path = DataUtils.get_cache_file_path(
                f"{comp_id}.pdb", output_dir
            )
            PdbBetaFactorHelper.create_pdb_with_beta_factors(
                pipeline_data, traj_idx, frame_idx,
                beta_factors, pdb_path
            )

            # Store in visualization data
            viz_data.add_pdb(comp_id, pdb_path)

        # Store in pipeline_data
        pipeline_data.structure_visualization_data[structure_viz_name] = viz_data

    def visualize_nglview_jupyter(
        self,
        pipeline_data: PipelineData,
        structure_viz_name: str,
        n_top_global: int = 3
    ) -> Tuple:
        """
        Create interactive NGLView widget for Jupyter notebooks.

        Creates 3D interactive visualization widget using NGLView with
        beta-factor gradient coloring and feature highlights. Requires
        PDFs to be created first via create_pdb_with_beta_factors().
        Automatically displays the widget in Jupyter and returns it.

        Warning
        -------
        - Only works in Jupyter notebook environment
        - Requires nglview and ipywidgets packages
        - When using PipelineManager, do NOT provide pipeline_data parameter

        Parameters
        ----------
        pipeline_data : PipelineData
            Pipeline data object (auto-injected by PipelineManager)
        structure_viz_name : str
            Name of visualization session (from create_pdb_with_beta_factors)
        n_top_global : int, default=3
            Number of global top features to highlight with licorice

        Returns
        -------
        Tuple[widgets.VBox, nv.NGLWidget]
            UI container (dropdown + checkboxes) and NGLView widget

        Raises
        ------
        RuntimeError
            If not running in Jupyter notebook environment
        KeyError
            If structure_viz_name not found in pipeline_data
        ImportError
            If nglview or ipywidgets not installed

        Examples
        --------
        >>> # First create PDFs
        >>> pipeline.structure_visualization.create_pdb_with_beta_factors(
        ...     "my_viz", "dt_analysis", n_top=10
        ... )
        >>> # Then visualize in Jupyter (automatically displayed)
        >>> ui, view = pipeline.structure_visualization.visualize_nglview_jupyter(
        ...     "my_viz", n_top_global=3
        ... )

        Notes
        -----
        - Beta-factor gradient: base color (important) → white (unimportant)
        - Dropdown allows switching between structures or "multiple" mode
        - Multi-view mode shows all structures with opacity checkboxes
        - Feature highlights use licorice representation
        - Widget is automatically displayed via IPython.display.display()
        """
        # Validate environment and data
        ValidationHelper.validate_jupyter_environment()
        viz_data = ValidationHelper.validate_visualization_data(
            pipeline_data, structure_viz_name
        )

        # Lazy import of Jupyter-specific dependencies
        # Neccessary to avoid import errors outside Jupyter
        from ..helpers.nglview_helper import NGLViewHelper
        from IPython.display import display

        # Get feature importance name from viz_data
        feature_importance_name = viz_data.feature_importance_name

        # Get feature importance data
        fi_data = pipeline_data.feature_importance_data[feature_importance_name]
        comp_data = pipeline_data.comparison_data[fi_data.comparison_name]

        # Prepare PDB info with colors
        pdb_info = self._prepare_pdb_info(
            viz_data, comp_data
        )

        # Get top features and colors
        top_features = TopFeaturesUtils.get_top_features_with_names(
            pipeline_data, fi_data, None, n_top_global
        )
        feature_colors = self._assign_feature_colors(top_features, n_top_global)

        # Create NGLView widget
        ui, view = NGLViewHelper.create_widget(
            pdb_info, top_features, feature_colors
        )

        # Automatically display in Jupyter
        display(ui, view)

        return ui, view

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

    def _prepare_pdb_info(
        self,
        viz_data: StructureVisualizationData,
        comp_data: Any
    ) -> Dict[str, Dict[str, str]]:
        """
        Prepare PDB info dictionary with paths and colors.

        Creates dictionary mapping sub-comparison names to PDB paths
        and assigned colors for visualization.

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
            - Values: {"path": pdb_path, "color": hex_color}
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
                    "path": pdb_path,
                    "color": structure_colors[i]
                }

        return pdb_info

    @staticmethod
    def _assign_feature_colors(
        top_features: List[Dict[str, Any]],
        n_features: int
    ) -> Dict[str, str]:
        """
        Assign distinct colors to top features.

        Creates color mapping for top features using visually distinct
        colors for highlighting in visualizations.

        Parameters
        ----------
        top_features : List[Dict[str, Any]]
            List of top feature dictionaries
        n_features : int
            Number of features to assign colors to

        Returns
        -------
        Dict[str, str]
            Mapping from feature name to HEX color string

        Examples
        --------
        >>> features = [
        ...     {"feature_name": "ALA_5_CA-GLU_10_CA"},
        ...     {"feature_name": "GLY_3_phi"}
        ... ]
        >>> colors = StructureVisualizationManager._assign_feature_colors(
        ...     features, 2
        ... )
        >>> print(colors["ALA_5_CA-GLU_10_CA"])  # e.g., "#1f77b4"
        """
        colors_dict = ColorUtils.generate_distinct_colors(n_features)

        feature_colors = {}
        for i, feature in enumerate(top_features[:n_features]):
            feature_name = feature.get("feature_name", f"feature_{i}")
            feature_colors[feature_name] = colors_dict[i]

        return feature_colors
