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
Validation helper for structure visualization operations.

This module provides validation utilities for structure visualization,
including Jupyter environment checks and visualization data validation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ...pipeline.entities.pipeline_data import PipelineData
    from ..entities.structure_visualization_data import StructureVisualizationData

from .environment_helper import EnvironmentHelper


class ValidationHelper:
    """
    Helper class for structure visualization validation.

    Provides static methods for validating environment requirements
    and data availability for structure visualization operations.

    Examples
    --------
    >>> ValidationHelper.validate_jupyter_environment()
    >>> viz_data = ValidationHelper.validate_visualization_data(
    ...     pipeline_data, "my_viz"
    ... )
    """

    @staticmethod
    def validate_jupyter_environment() -> None:
        """
        Validate that code is running in Jupyter environment.

        Parameters
        ----------
        None

        Returns
        -------
        None
            Returns normally if running in Jupyter environment

        Raises
        ------
        RuntimeError
            If not running in Jupyter notebook environment

        Examples
        --------
        >>> ValidationHelper.validate_jupyter_environment()
        RuntimeError: NGLView visualization is only available in Jupyter notebooks...

        Notes
        -----
        Uses EnvironmentHelper.is_jupyter_environment() to check
        environment. Provides helpful error message for terminal usage.
        """
        if not EnvironmentHelper.is_jupyter_environment():
            raise RuntimeError(
                "NGLView visualization is only available in Jupyter notebooks. "
                "For terminal/script usage, please use PyMOL visualization instead:\n"
                "  pipeline.structure_visualization.visualize_pymol(...)"
            )

    @staticmethod
    def validate_visualization_data(
        pipeline_data: PipelineData,
        structure_viz_name: str
    ) -> StructureVisualizationData:
        """
        Validate visualization data exists and return it.

        Parameters
        ----------
        pipeline_data : PipelineData
            Pipeline data object containing visualization data
        structure_viz_name : str
            Name of visualization session to validate

        Returns
        -------
        StructureVisualizationData
            Validated visualization data object

        Raises
        ------
        KeyError
            If structure_visualization_data attribute not found
        KeyError
            If specific visualization session name not found

        Examples
        --------
        >>> viz_data = ValidationHelper.validate_visualization_data(
        ...     pipeline_data, "my_viz"
        ... )

        >>> # Raises KeyError if not found
        >>> viz_data = ValidationHelper.validate_visualization_data(
        ...     pipeline_data, "nonexistent"
        ... )
        KeyError: Visualization 'nonexistent' not found...

        Notes
        -----
        Provides detailed error messages listing available visualizations
        if the requested one is not found.
        """
        # Check if structure_visualization_data exists
        if not hasattr(pipeline_data, 'structure_visualization_data'):
            raise KeyError(
                "No structure visualization data found. "
                "Run create_pdb_with_beta_factors() first."
            )

        # Get specific visualization data
        viz_data = pipeline_data.structure_visualization_data.get(
            structure_viz_name
        )

        if viz_data is None:
            available = list(pipeline_data.structure_visualization_data.keys())
            raise KeyError(
                f"Visualization '{structure_viz_name}' not found. "
                f"Available: {available}"
            )

        return viz_data

    @staticmethod
    def validate_terminal_environment() -> None:
        """
        Validate that code is running in terminal, NOT Jupyter.

        Checks if code is running in Jupyter notebook environment and
        raises error if true. PyMOL visualization requires terminal
        execution.

        Parameters
        ----------
        None

        Returns
        -------
        None
            Returns normally if NOT in Jupyter environment

        Raises
        ------
        RuntimeError
            If running in Jupyter notebook environment

        Examples
        --------
        >>> ValidationHelper.validate_terminal_environment()
        RuntimeError: PyMOL visualization requires terminal execution...

        Notes
        -----
        Uses EnvironmentHelper.is_jupyter_environment() to check
        environment. Provides helpful message suggesting NGLView
        for Jupyter usage.
        """
        if EnvironmentHelper.is_jupyter_environment():
            raise RuntimeError(
                "PyMOL visualization requires terminal/script execution. "
                "Use visualize_nglview_jupyter() in Jupyter notebooks instead."
            )

    @staticmethod
    def validate_pymol_available() -> None:
        """
        Validate that PyMOL Python module is available.

        Checks if pymol module can be imported and raises error if not.
        Provides installation instructions for conda.

        Parameters
        ----------
        None

        Returns
        -------
        None
            Returns normally if pymol is available

        Raises
        ------
        ImportError
            If pymol Python module cannot be imported

        Examples
        --------
        >>> ValidationHelper.validate_pymol_available()
        ImportError: PyMOL Python module not found...

        Notes
        -----
        Uses EnvironmentHelper.is_pymol_available() to check
        availability. Suggests conda installation command if missing.
        """
        if not EnvironmentHelper.is_pymol_available():
            raise ImportError(
                "PyMOL Python module not found. Install with:\n"
                "  conda install -c conda-forge pymol-open-source"
            )

    @staticmethod
    def validate_data_selectors(
        pipeline_data: PipelineData,
        data_selectors: list
    ) -> None:
        """
        Validate that data selectors exist in pipeline data.

        Checks that all provided data selector names are present in
        pipeline_data.data_selector_data and raises errors for missing
        or empty selector lists.

        Parameters
        ----------
        pipeline_data : PipelineData
            Pipeline data object containing data selector data
        data_selectors : list
            List of data selector names to validate

        Returns
        -------
        None
            Returns normally if all data selectors are valid

        Raises
        ------
        ValueError
            If no data selectors provided or selector doesn't exist

        Examples
        --------
        >>> ValidationHelper.validate_data_selectors(
        ...     pipeline_data, ["cluster_0", "cluster_1"]
        ... )

        >>> # Raises ValueError if empty
        >>> ValidationHelper.validate_data_selectors(pipeline_data, [])
        ValueError: At least one data selector is required

        >>> # Raises ValueError if not found
        >>> ValidationHelper.validate_data_selectors(
        ...     pipeline_data, ["nonexistent"]
        ... )
        ValueError: Data selector 'nonexistent' not found...

        Notes
        -----
        This method is used by StructureVizFeatureService to validate
        data selectors before creating representative PDFs.
        """
        if not data_selectors:
            raise ValueError("At least one data selector is required")

        for ds_name in data_selectors:
            if ds_name not in pipeline_data.data_selector_data:
                raise ValueError(
                    f"Data selector '{ds_name}' not found in pipeline_data"
                )
