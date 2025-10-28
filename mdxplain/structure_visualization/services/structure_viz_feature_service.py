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
Service for feature-selector-based structure visualization.

This service provides methods for creating and visualizing molecular
structures based on feature selectors and data selectors without
requiring feature importance analysis.
"""

from __future__ import annotations

import numpy as np
from typing import List, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from ...pipeline.entities.pipeline_data import PipelineData
    from ..managers.structure_visualization_manager import (
        StructureVisualizationManager,
    )

from ..entities.structure_visualization_data import StructureVisualizationData
from ..helpers.pdb_beta_factor_helper import PdbBetaFactorHelper
from ..helpers.validation_helper import ValidationHelper
from ..helpers.visualization_data_helper import VisualizationDataHelper
from ...utils.data_utils import DataUtils


class StructureVizFeatureService:
    """
    Service for feature-selector-based structure visualization.

    Provides methods for creating PDB files from data selector centroids
    with optional feature highlighting from feature selectors. Does not
    require feature importance analysis.

    Examples
    --------
    >>> # Via PipelineManager
    >>> pipeline.structure_visualization.feature.create_representative_pdbs(
    ...     "my_viz",
    ...     data_selectors=["cluster_0", "cluster_1"],
    ...     selector_centroid="coords_all",
    ...     selector_features="important_distances"
    ... )
    >>> ui, view = pipeline.structure_visualization.feature.visualize_nglview_jupyter(
    ...     "my_viz"
    ... )
    """

    def __init__(
        self,
        manager: StructureVisualizationManager,
        pipeline_data: PipelineData
    ):
        """
        Initialize feature-based visualization service.

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

    def create_representative_pdbs(
        self,
        structure_viz_name: str,
        data_selectors: List[str],
        selector_centroid: str,
        selector_features: str = None,
        output_dir: str = None
    ) -> None:
        """
        Create PDB files from data selector centroids.

        Generates PDB files for each data selector using centroid frames
        calculated from selector_centroid. Beta-factors are set to 0.0
        (uniform). Optional feature highlighting from selector_features.

        Parameters
        ----------
        structure_viz_name : str
            Name for this visualization session
        data_selectors : List[str]
            List of data selector names (min 1 required)
        selector_centroid : str
            Feature selector name for centroid calculation (REQUIRED)
        selector_features : str, optional
            Feature selector for visualization highlights (default None)
        output_dir : str, optional
            Output directory for PDB files (defaults to cache_dir/structure_viz)

        Returns
        -------
        None
            Stores PDB paths in pipeline_data.structure_visualization_data

        Examples
        --------
        >>> # With feature highlighting
        >>> pipeline.structure_visualization.feature.create_representative_pdbs(
        ...     "my_viz",
        ...     data_selectors=["cluster_0", "cluster_1", "cluster_2"],
        ...     selector_centroid="coords_all",
        ...     selector_features="important_distances"
        ... )
        >>> # Without features (only structures)
        >>> pipeline.structure_visualization.feature.create_representative_pdbs(
        ...     "my_viz",
        ...     data_selectors=["cluster_0", "cluster_1"],
        ...     selector_centroid="coords_all"
        ... )

        Notes
        -----
        - Creates one PDB per data selector
        - Beta-factors always 0.0 (uniform thickness)
        - Centroids calculated using selector_centroid
        - Features from selector_features shown as highlights (if provided)
        """
        # Use provided output_dir or default to manager's output_dir
        if output_dir is None:
            output_dir = self._manager.output_dir

        # Validate data_selectors
        ValidationHelper.validate_data_selectors(
            self._pipeline_data, data_selectors
        )

        # Create visualization data entity
        viz_data = StructureVisualizationData(
            structure_viz_name,
            selector_centroid=selector_centroid,
            selector_features=selector_features
        )

        # Process each data selector
        for ds_name in data_selectors:
            # Get centroid frame
            traj_idx, frame_idx = self._pipeline_data.get_centroid_frame(
                selector_centroid, ds_name
            )

            # Get topology for beta-factors
            topology = PdbBetaFactorHelper.get_topology(
                self._pipeline_data, traj_idx
            )

            # Create beta-factors (all 0.0)
            beta_factors = np.zeros(topology.n_atoms)

            # Create PDB
            pdb_path = DataUtils.get_cache_file_path(
                f"{ds_name}.pdb", output_dir
            )
            PdbBetaFactorHelper.create_pdb_with_beta_factors(
                self._pipeline_data, traj_idx, frame_idx,
                beta_factors, pdb_path
            )

            # Store in visualization data
            viz_data.add_pdb(ds_name, pdb_path)

        # Extract and store features if selector_features provided
        if selector_features is not None:
            features = VisualizationDataHelper.extract_features_from_selector(
                self._pipeline_data, selector_features
            )
            viz_data.add_feature_info(features)

        # Store in pipeline_data
        self._pipeline_data.structure_visualization_data[structure_viz_name] = viz_data

    def visualize_nglview_jupyter(
        self,
        structure_viz_name: str,
        feature_own_color: bool = True
    ) -> Tuple:
        """
        Create interactive NGLView widget for Jupyter notebooks.

        Creates 3D interactive visualization widget using NGLView with
        uniform structure coloring and optional feature highlights. Requires
        PDFs to be created first via create_representative_pdbs().
        Automatically displays the widget in Jupyter and returns it.

        Warning
        -------
        - Only works in Jupyter notebook environment
        - Requires nglview and ipywidgets packages

        Parameters
        ----------
        structure_viz_name : str
            Name of visualization session (from create_representative_pdbs)
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
        >>> pipeline.structure_visualization.feature.create_representative_pdbs(
        ...     "my_viz",
        ...     data_selectors=["cluster_0", "cluster_1"],
        ...     selector_centroid="coords_all",
        ...     selector_features="distances"
        ... )
        >>> # Then visualize in Jupyter (automatically displayed)
        >>> ui, view = pipeline.structure_visualization.feature.visualize_nglview_jupyter(
        ...     "my_viz"
        ... )

        Notes
        -----
        - All structures have uniform thickness (beta=0.0)
        - Feature highlights use licorice representation
        - Dropdown allows switching between structures
        - Widget is automatically displayed via IPython.display.display()
        """
        # Validate environment and data
        ValidationHelper.validate_jupyter_environment()
        viz_data = ValidationHelper.validate_visualization_data(
            self._pipeline_data, structure_viz_name
        )

        # Lazy import of Jupyter-specific dependencies
        from ..helpers.nglview_helper import NGLViewHelper
        from IPython.display import display

        # Prepare PDB info with colors
        pdb_info = VisualizationDataHelper.prepare_pdb_info_from_viz_data(
            viz_data
        )

        # Get features and colors
        features = viz_data.get_feature_info()
        feature_colors = VisualizationDataHelper.assign_feature_colors(
            features, len(features)
        )

        # Create NGLView widget
        ui, view = NGLViewHelper.create_widget(
            pdb_info, features, feature_colors, feature_own_color,
            viz_name=structure_viz_name
        )

        # Automatically display in Jupyter
        display(ui, view)

        return ui, view

    def create_pymol_script(
        self,
        structure_viz_name: str,
        output_dir: str = None,
        feature_own_color: bool = True
    ) -> str:
        """
        Create PyMOL script (.pml) for structure visualization.

        Generates PyMOL script with uniform cartoons (beta=0.0) and
        optional feature objects as toggleable stick representations.

        Parameters
        ----------
        structure_viz_name : str
            Name of visualization session (from create_representative_pdbs)
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
        >>> pipeline.structure_visualization.feature.create_representative_pdbs(
        ...     "my_viz",
        ...     data_selectors=["cluster_0", "cluster_1"],
        ...     selector_centroid="coords_all",
        ...     selector_features="distances"
        ... )
        >>> # Then create PyMOL script
        >>> script_path = pipeline.structure_visualization.feature.create_pymol_script(
        ...     "my_viz"
        ... )
        >>> print(f"Script saved to: {script_path}")

        Notes
        -----
        - Focus groups named: all_focus_struct_{name}
        - Each group contains own structure + others for context
        - Only first focus group enabled by default
        - Toggle groups in PyMOL GUI to switch focus
        - Script saved as: output_dir/structure_viz_name.pml
        - Uniform cartoons (no beta-factor variation)
        """
        # Get output_dir
        if output_dir is None:
            output_dir = self._manager.output_dir

        # Get viz_data
        viz_data = ValidationHelper.validate_visualization_data(
            self._pipeline_data, structure_viz_name
        )

        # Lazy import PyMOL helper
        from ..helpers.pymol_script_generator import PyMolScriptGenerator

        # Prepare PDB info
        pdb_info = VisualizationDataHelper.prepare_pdb_info_from_viz_data(
            viz_data
        )

        # Get features + colors
        features = viz_data.get_feature_info()
        feature_colors = VisualizationDataHelper.assign_feature_colors(
            features, len(features)
        )

        # Generate script (use_putty=False for uniform beta-factors)
        script_content = PyMolScriptGenerator.generate_script(
            pdb_info, features, feature_colors, feature_own_color,
            use_putty=False
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
            Name of visualization session (from create_representative_pdbs)
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
        >>> pipeline.structure_visualization.feature.create_representative_pdbs(
        ...     "my_viz",
        ...     data_selectors=["cluster_0", "cluster_1"],
        ...     selector_centroid="coords_all"
        ... )
        >>> # Then open in PyMOL (terminal only!)
        >>> pipeline.structure_visualization.feature.visualize_pymol("my_viz")

        Notes
        -----
        - PyMOL opens with all structures and features visible
        - Toggle structures: enable/disable struct_<name>
        - Toggle features: enable/disable feat_<name>
        - Uniform cartoons (no beta-factor variation)
        - Script created if not already exists
        """
        # Validate terminal environment
        ValidationHelper.validate_terminal_environment()

        # Validate PyMOL available
        ValidationHelper.validate_pymol_available()

        # Create script
        script_path = self.create_pymol_script(
            structure_viz_name,
            feature_own_color=feature_own_color
        )

        # Import pymol lazy cause optional dependency
        import pymol

        # Launch PyMOL in quiet mode
        pymol.finish_launching(['pymol', '-q'])

        # Load script
        pymol.cmd.do(f"@{script_path}")
