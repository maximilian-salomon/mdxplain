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
Service for feature-importance-based structure visualization.

This service provides methods for creating and visualizing molecular
structures with beta-factors derived from feature importance analysis.
"""

from __future__ import annotations

from typing import Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from ...pipeline.entities.pipeline_data import PipelineData
    from ..managers.structure_visualization_manager import (
        StructureVisualizationManager,
    )

from ..entities.structure_visualization_data import StructureVisualizationData
from ..helpers.pdb_beta_factor_helper import PdbBetaFactorHelper
from ..helpers.validation_helper import ValidationHelper
from ..helpers.visualization_data_helper import VisualizationDataHelper
from ..helpers.pymol_script_generator import PyMolScriptGenerator
from ...utils.top_features_utils import TopFeaturesUtils
from ...utils.data_utils import DataUtils


class StructureVizFeatureImportanceService:
    """
    Service for feature-importance-based structure visualization.

    Provides methods for creating PDB files with beta-factors derived
    from feature importance analysis and visualizing them with NGLView
    (Jupyter) or PyMOL (terminal/script).

    Examples
    --------
    >>> # Via PipelineManager
    >>> pipeline.structure_visualization.feature_importance.create_pdb_with_beta_factors(
    ...     "my_viz", "dt_analysis", n_top=10
    ... )
    >>> ui, view = pipeline.structure_visualization.feature_importance.visualize_nglview_jupyter(
    ...     "my_viz", n_top_global=3
    ... )
    """

    def __init__(
        self,
        manager: StructureVisualizationManager,
        pipeline_data: PipelineData
    ):
        """
        Initialize feature importance visualization service.

        Parameters
        ----------
        manager : StructureVisualizationManager
            Parent manager instance
        pipeline_data : PipelineData
            Pipeline data object (injected by AutoInjectProxy)

        Returns
        -------
        None
            Initializes service instance
        """
        self._manager = manager
        self._pipeline_data = pipeline_data

    def create_pdb_with_beta_factors(
        self,
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

        Parameters
        ----------
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
        >>> pipeline.structure_visualization.feature_importance.create_pdb_with_beta_factors(
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
            output_dir = self._manager.output_dir

        fi_data = self._pipeline_data.feature_importance_data[feature_importance_name]
        comp_data = self._pipeline_data.comparison_data[fi_data.comparison_name]

        # Create visualization data entity
        viz_data = StructureVisualizationData(structure_viz_name, feature_importance_name)

        # Process each sub-comparison
        for sub_comp in comp_data.sub_comparisons:
            comp_id = sub_comp["name"]

            # Get representative frame
            traj_idx, frame_idx = fi_data.get_representative_frame(
                self._pipeline_data, comp_id, representative_mode,
                n_top, self._manager.use_memmap, self._manager.chunk_size
            )

            # Calculate beta-factors
            beta_factors = self._manager._calculate_beta_factors(
                self._pipeline_data, fi_data, comp_id, traj_idx, n_top
            )

            # Create PDB
            pdb_path = DataUtils.get_cache_file_path(
                f"{comp_id}.pdb", output_dir
            )
            PdbBetaFactorHelper.create_pdb_with_beta_factors(
                self._pipeline_data, traj_idx, frame_idx,
                beta_factors, pdb_path
            )

            # Store in visualization data
            viz_data.add_pdb(comp_id, pdb_path)

        # Store in pipeline_data
        self._pipeline_data.structure_visualization_data[structure_viz_name] = viz_data

    def visualize_nglview_jupyter(
        self,
        structure_viz_name: str,
        n_top_global: int = 3,
        feature_own_color: bool = True
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

        Parameters
        ----------
        structure_viz_name : str
            Name of visualization session (from create_pdb_with_beta_factors)
        n_top_global : int, default=3
            Number of global top features to highlight with licorice
        feature_own_color : bool, default=True
            If True, features use their own color from feature_colors.
            If False, features use the color of their structure.

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
        >>> pipeline.structure_visualization.feature_importance.create_pdb_with_beta_factors(
        ...     "my_viz", "dt_analysis", n_top=10
        ... )
        >>> # Then visualize in Jupyter (automatically displayed)
        >>> ui, view = pipeline.structure_visualization.feature_importance.visualize_nglview_jupyter(
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
            self._pipeline_data, structure_viz_name
        )

        # Lazy import of Jupyter-specific dependencies
        # Neccessary to avoid import errors outside Jupyter
        from ..helpers.nglview_helper import NGLViewHelper
        from IPython.display import display

        # Get feature importance name from viz_data
        feature_importance_name = viz_data.feature_importance_name

        # Get feature importance data
        fi_data = self._pipeline_data.feature_importance_data[feature_importance_name]
        comp_data = self._pipeline_data.comparison_data[fi_data.comparison_name]

        # Prepare PDB info with colors
        pdb_info = VisualizationDataHelper.prepare_pdb_info_from_comp_data(
            viz_data, comp_data
        )

        # Get top features and colors
        top_features = TopFeaturesUtils.get_top_features_with_names(
            self._pipeline_data, fi_data, None, n_top_global
        )
        feature_colors = VisualizationDataHelper.assign_feature_colors(
            top_features, n_top_global
        )

        # Create NGLView widget
        ui, view = NGLViewHelper.create_widget(
            pdb_info, top_features, feature_colors, feature_own_color,
            viz_name=structure_viz_name
        )

        # Automatically display in Jupyter
        display(ui, view)

        return ui, view

    def create_pymol_script(
        self,
        structure_viz_name: str,
        n_top_global: int = 3,
        output_dir: str = None,
        feature_own_color: bool = True
    ) -> str:
        """
        Create PyMOL script (.pml) for structure visualization.

        Generates PyMOL script with putty cartoons (thickness from
        beta-factor), color gradients (base_color → white), and
        feature objects as toggleable stick representations.

        Parameters
        ----------
        structure_viz_name : str
            Name of visualization session (from create_pdb_with_beta_factors)
        n_top_global : int, default=3
            Number of top features to highlight
        output_dir : str, optional
            Output directory (defaults to cache_dir/structure_viz)
        feature_own_color : bool, default=True
            If True, features use their own color from feature_colors.
            If False, features use the color of their structure.

        Returns
        -------
        str
            Path to created .pml script file

        Examples
        --------
        >>> # First create PDFs
        >>> pipeline.structure_visualization.feature_importance.create_pdb_with_beta_factors(
        ...     "my_viz", "dt_analysis", n_top=10
        ... )
        >>> # Then create PyMOL script
        >>> script_path = pipeline.structure_visualization.feature_importance.create_pymol_script(
        ...     "my_viz", n_top_global=3
        ... )
        >>> print(f"Script saved to: {script_path}")

        Notes
        -----
        - Focus groups named: all_focus_struct_{name}
        - Each group contains:
          * Own structure + features (opak)
          * Other structures + features (80% transparent, context)
        - Only first focus group enabled by default
        - Toggle groups in PyMOL GUI to switch focus
        - Script saved as: output_dir/structure_viz_name.pml
        - Putty cartoons: thickness controlled by beta-factor
        - Color gradient: base_color (low B) → white (high B)
        - Stick colors independent of cartoon (beta-factor preserved)
        """
        # Get output_dir
        if output_dir is None:
            output_dir = self._manager.output_dir

        # Get viz_data
        viz_data = ValidationHelper.validate_visualization_data(
            self._pipeline_data, structure_viz_name
        )

        # Get feature importance data
        fi_data = self._pipeline_data.feature_importance_data[viz_data.feature_importance_name]
        comp_data = self._pipeline_data.comparison_data[fi_data.comparison_name]

        # Prepare PDB info
        pdb_info = VisualizationDataHelper.prepare_pdb_info_from_comp_data(
            viz_data, comp_data
        )

        # Get top features + colors
        top_features = TopFeaturesUtils.get_top_features_with_names(
            self._pipeline_data, fi_data, None, n_top_global
        )
        feature_colors = VisualizationDataHelper.assign_feature_colors(
            top_features, n_top_global
        )

        # Generate script (use_putty=True for beta-factor variation)
        script_content = PyMolScriptGenerator.generate_script(
            pdb_info, top_features, feature_colors, feature_own_color,
            use_putty=True
        )

        # Save script
        script_path = DataUtils.get_cache_file_path(
            f"{structure_viz_name}.pml", output_dir
        )
        with open(script_path, 'w') as f:
            f.write(script_content)

        return script_path

    def visualize_pymol(
        self,
        structure_viz_name: str,
        n_top_global: int = 3,
        feature_own_color: bool = True
    ) -> None:
        """
        Open PyMOL with structure visualization.

        Validates environment (terminal, PyMOL available), creates
        script if needed, and launches PyMOL with pymol Python module.

        Warning
        -------
        - Only works in terminal/script, NOT in Jupyter notebooks
        - Requires pymol Python module (conda install pymol-open-source)

        Parameters
        ----------
        structure_viz_name : str
            Name of visualization session (from create_pdb_with_beta_factors)
        n_top_global : int = 3
            Number of top features to highlight
        feature_own_color : bool, default=True
            If True, features use their own color from feature_colors.
            If False, features use the color of their structure.

        Returns
        -------
        None
            Opens PyMOL with visualization

        Raises
        ------
        RuntimeError
            If running in Jupyter notebook
        ImportError
            If pymol Python module not available

        Examples
        --------
        >>> # First create PDFs
        >>> pipeline.structure_visualization.feature_importance.create_pdb_with_beta_factors(
        ...     "my_viz", "dt_analysis", n_top=10
        ... )
        >>> # Then open in PyMOL (terminal only!)
        >>> pipeline.structure_visualization.feature_importance.visualize_pymol(
        ...     "my_viz", n_top_global=3
        ... )

        Notes
        -----
        - PyMOL opens with all structures and features visible
        - Toggle structures: enable/disable struct_<name>
        - Toggle features: enable/disable feat_<name>
        - Putty cartoons: thickness controlled by beta-factor
        - Color gradient: base_color (low B) → white (high B)
        - Script created if not already exists
        """
        # Validate terminal environment
        ValidationHelper.validate_terminal_environment()

        # Validate PyMOL available
        ValidationHelper.validate_pymol_available()

        # Create script
        script_path = self.create_pymol_script(
            structure_viz_name, n_top_global,
            feature_own_color=feature_own_color
        )

        # Import pymol lazy cause optional dependency
        import pymol

        # Launch PyMOL in quiet mode
        pymol.finish_launching(['pymol', '-q'])

        # Load script
        pymol.cmd.do(f"@{script_path}")
