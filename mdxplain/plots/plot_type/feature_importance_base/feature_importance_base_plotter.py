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
Base plotter for feature importance visualization.

Provides shared functionality for violin and density plotters to eliminate
code duplication while maintaining clean separation of concerns.
"""

from typing import Optional, Dict, List, Tuple
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec
from ...helper.grid_layout_helper import GridLayoutHelper
from ...helper.title_legend_helper import TitleLegendHelper
from ...helper.validation_helper import ValidationHelper


class FeatureImportanceBasePlotter:
    """
    Base class for feature importance plotters.

    Provides shared functionality for violin and density plotters including
    mode validation, feature flattening, figure creation, and subplot
    measurements.

    Examples
    --------
    >>> class ViolinPlotter(FeatureImportanceBasePlotter):
    ...     def plot(self, ...):
    ...         # Use inherited methods
    ...         mode_type, mode_name = self._validate_and_determine_mode(...)
    ...         fig = self._create_figure(n_rows, n_cols)
    """

    def __init__(self, pipeline_data, cache_dir: str) -> None:
        """
        Initialize base plotter.

        Parameters
        ----------
        pipeline_data : PipelineData
            Pipeline data container
        cache_dir : str
            Cache directory path

        Returns
        -------
        None
            Initializes FeatureImportanceBasePlotter instance
        """
        self.pipeline_data = pipeline_data
        self.cache_dir = cache_dir

    def _validate_and_determine_mode(
        self,
        feature_importance_name: Optional[str],
        feature_selector: Optional[str],
        data_selectors: Optional[List[str]],
    ) -> Tuple[str, str]:
        """
        Validate parameters and determine operational mode.

        Checks for mode conflicts, validates required parameters,
        and determines whether Feature Importance or Manual mode.

        Parameters
        ----------
        feature_importance_name : str, optional
            Feature importance analysis name
        feature_selector : str, optional
            Feature selector name
        data_selectors : List[str], optional
            DataSelector names

        Returns
        -------
        mode_type : str
            "feature_importance" or "manual"
        mode_name : str
            Name for this mode (fi_name or feature_selector_name)

        Raises
        ------
        ValueError
            If both modes specified or required parameters missing

        Examples
        --------
        >>> # Feature Importance mode
        >>> mode_type, mode_name = plotter._validate_and_determine_mode(
        ...     "tree_analysis", None, None
        ... )
        >>> print(mode_type, mode_name)
        feature_importance tree_analysis

        >>> # Manual mode
        >>> mode_type, mode_name = plotter._validate_and_determine_mode(
        ...     None, "my_selector", ["cluster_0", "cluster_1"]
        ... )
        >>> print(mode_type, mode_name)
        manual my_selector
        """
        if feature_importance_name is not None:
            self._validate_exclusive_fi_mode(feature_selector, data_selectors)
            return "feature_importance", feature_importance_name

        self._validate_manual_mode_params(feature_selector, data_selectors)
        return "manual", feature_selector

    @staticmethod
    def _validate_exclusive_fi_mode(
        feature_selector: Optional[str],
        data_selectors: Optional[List[str]]
    ) -> None:
        """
        Validate Feature Importance mode is exclusive.

        Parameters
        ----------
        feature_selector : str, optional
            Feature selector name (must be None for FI mode)
        data_selectors : List[str], optional
            DataSelector names (must be None for FI mode)

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If manual mode parameters also specified
        """
        if feature_selector is not None or data_selectors is not None:
            raise ValueError(
                "Cannot mix Feature Importance and Manual modes. "
                "Provide either feature_importance_name OR "
                "(feature_selector + data_selectors)."
            )

    @staticmethod
    def _validate_manual_mode_params(
        feature_selector: Optional[str],
        data_selectors: Optional[List[str]]
    ) -> None:
        """
        Validate Manual mode has required parameters.

        Parameters
        ----------
        feature_selector : str, optional
            Feature selector name (required for Manual mode)
        data_selectors : List[str], optional
            DataSelector names (required for Manual mode)

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If either required parameter is missing
        """
        if feature_selector is None or data_selectors is None:
            raise ValueError(
                "Manual mode requires both feature_selector and data_selectors. "
                "Provide these parameters or use feature_importance_name for "
                "Feature Importance mode."
            )

    def _flatten_features(
        self,
        plot_data: Dict[str, Dict[str, Dict[str, any]]]
    ) -> List[Tuple[str, str]]:
        """
        Flatten nested structure to list of (feature_type, feature_name).

        Converts three-level nested dict to flat list of features. Each entry
        represents one subplot that will contain visualizations for all
        DataSelectors of that feature.

        Parameters
        ----------
        plot_data : Dict[str, Dict[str, Dict[str, any]]]
            Nested structure {feat_type: {feat_name: {selector: values}}}

        Returns
        -------
        List[Tuple[str, str]]
            List of (feature_type, feature_name) tuples

        Examples
        --------
        >>> features = plotter._flatten_features(plot_data)
        >>> print(len(features))  # e.g., 10 (10 features)
        >>> print(features[0])
        ('distances', 'ALA26-PRO35')
        """
        all_features = []
        for feat_type, features in plot_data.items():
            for feat_name in features.keys():
                all_features.append((feat_type, feat_name))
        return all_features

    def _create_figure(self, n_rows: int, n_cols: int) -> Figure:
        """
        Create figure with appropriate size.

        Calculates figure dimensions based on grid layout, ensuring minimum
        size while scaling with number of subplots.

        Parameters
        ----------
        n_rows : int
            Number of rows in grid
        n_cols : int
            Number of columns in grid

        Returns
        -------
        Figure
            Created matplotlib figure

        Examples
        --------
        >>> fig = plotter._create_figure(n_rows=3, n_cols=4)
        >>> print(fig.get_figwidth(), fig.get_figheight())
        20 12
        """
        fig_width = max(10, n_cols * 5)
        fig_height = max(8, n_rows * 4)
        return plt.figure(figsize=(fig_width, fig_height))

    def _measure_subplot_width(
        self,
        fig: Figure,
        n_rows: int,
        n_cols: int,
        wspace: float,
        hspace: float
    ) -> float:
        """
        Measure actual subplot width by creating dummy GridSpec.

        Creates temporary GridSpec and axes to measure the actual width
        of a subplot in inches. This width is used to calculate appropriate
        character limits for title wrapping.

        Parameters
        ----------
        fig : Figure
            Matplotlib figure
        n_rows : int
            Number of rows in grid
        n_cols : int
            Number of columns in grid
        wspace : float
            GridSpec wspace parameter (horizontal spacing)
        hspace : float
            GridSpec hspace parameter (vertical spacing)

        Returns
        -------
        float
            Subplot width in inches

        Examples
        --------
        >>> fig = plt.figure(figsize=(20, 16))
        >>> width = plotter._measure_subplot_width(fig, 3, 4, 0.4, 0.25)
        >>> print(f"Subplot width: {width:.2f} inches")
        Subplot width: 4.25 inches
        """
        # Create temporary GridSpec with dummy top value
        temp_gs = GridSpec(
            n_rows, n_cols, figure=fig,
            hspace=hspace,
            wspace=wspace,
            top=0.9  # Temporary value for measurement
        )

        # Create dummy axes for first subplot
        dummy_ax = fig.add_subplot(temp_gs[0, 0])

        # Get position and calculate width in inches
        pos = dummy_ax.get_position()
        width_inches = pos.width * fig.get_figwidth()

        # Remove dummy axes
        fig.delaxes(dummy_ax)

        return width_inches

    def _setup_layout_and_figure(
        self,
        all_features: List[Tuple[str, str]],
        max_cols: int
    ) -> Tuple[List, int, int, Figure]:
        """
        Setup grid layout and create figure.

        Parameters
        ----------
        all_features : List[Tuple[str, str]]
            List of (feature_type, feature_name) tuples
        max_cols : int
            Maximum columns per row

        Returns
        -------
        Tuple[List, int, int, Figure]
            (layout, n_rows, n_cols, fig)
        """
        layout, n_rows, n_cols = GridLayoutHelper.compute_uniform_grid_layout(
            len(all_features), max_cols
        )
        fig = self._create_figure(n_rows, n_cols)
        return layout, n_rows, n_cols, fig

    def _configure_plot_spacing(
        self,
        all_features: List[Tuple[str, str]],
        metadata_map: Dict[str, Dict[str, Dict[str, any]]],
        long_labels: bool
    ) -> Tuple[bool, float, float]:
        """
        Configure plot spacing based on feature types.

        Parameters
        ----------
        all_features : List[Tuple[str, str]]
            List of (feature_type, feature_name) tuples
        metadata_map : Dict[str, Dict[str, Dict[str, any]]]
            Feature metadata map
        long_labels : bool
            Whether using long labels for discrete features

        Returns
        -------
        Tuple[bool, float, float]
            (has_discrete, wspace, hspace)
        """
        has_discrete = ValidationHelper.has_discrete_features(
            all_features, metadata_map
        )

        wspace = 0.8 if (long_labels and has_discrete) else 0.4
        hspace = 0.25
        return has_discrete, wspace, hspace

    def _create_gridspec_with_title(
        self,
        fig: Figure,
        n_rows: int,
        n_cols: int,
        wspace: float,
        hspace: float,
        title: Optional[str],
        default_title: str
    ) -> Tuple[GridSpec, str, float]:
        """
        Create GridSpec with calculated top spacing for title.

        Parameters
        ----------
        fig : Figure
            Matplotlib figure
        n_rows : int
            Number of rows
        n_cols : int
            Number of columns
        wspace : float
            Horizontal spacing
        hspace : float
            Vertical spacing
        title : str, optional
            Custom title
        default_title : str
            Default title if custom not provided

        Returns
        -------
        Tuple[GridSpec, str, float]
            (gs, wrapped_title, top)
        """
        subplot_width = self._measure_subplot_width(
            fig, n_rows, n_cols, wspace, hspace
        )
        max_chars = TitleLegendHelper.compute_title_max_chars_from_width(
            subplot_width
        )
        title_text = title if title else default_title
        wrapped_title, title_height = TitleLegendHelper.estimate_title_height(
            title_text, max_chars_per_line=max_chars
        )
        gap_to_plots = 0.4
        legend_height = 0.3
        title_legend_space = title_height + gap_to_plots + legend_height
        top = 1.0 - (title_legend_space / fig.get_figheight())
        gs = GridSpec(
            n_rows, n_cols, figure=fig,
            hspace=hspace, wspace=wspace, top=top
        )
        return gs, wrapped_title, top

    def _add_title_and_legend_positioned(
        self,
        fig: Figure,
        wrapped_title: str,
        top: float,
        rightmost_ax_first_row,
        data_selector_colors: Dict[str, str],
        legend_title: Optional[str],
        legend_labels: Optional[Dict[str, str]],
        active_threshold: Optional[float]
    ) -> None:
        """
        Add title and legend with calculated positions.

        Parameters
        ----------
        fig : Figure
            Figure to add title/legend to
        wrapped_title : str
            Pre-wrapped title text
        top : float
            Top position for GridSpec
        rightmost_ax_first_row
            Rightmost axes in first row
        data_selector_colors : Dict[str, str]
            DataSelector color mapping
        legend_title : str, optional
            Custom legend title
        legend_labels : Dict[str, str], optional
            Custom legend labels
        active_threshold : float, optional
            Threshold value for legend

        Returns
        -------
        None
            Modifies fig in place
        """
        title_offset_from_top = 0.15
        title_y = 1.0 - (title_offset_from_top / fig.get_figheight())
        TitleLegendHelper.add_title(fig, wrapped_title, title_y=title_y)

        pos = rightmost_ax_first_row.get_position()
        gap_inches = 0.1
        legend_x = pos.x1 + (gap_inches / fig.get_figwidth())
        legend_y = top + 0.01

        TitleLegendHelper.add_legend(
            fig, data_selector_colors,
            legend_title, legend_labels,
            contact_threshold=active_threshold,
            legend_x=legend_x, legend_y=legend_y
        )

    def _save_figure(
        self,
        fig: Figure,
        filename: Optional[str],
        mode_type: str,
        mode_name: str,
        n_top: int,
        file_format: str,
        dpi: int,
        prefix: str
    ) -> None:
        """
        Save figure to file.

        Generates automatic filename if not provided, adds file extension
        if missing, and saves with specified resolution.

        Parameters
        ----------
        fig : Figure
            Figure to save
        filename : str, optional
            Custom filename. If None, generates automatic filename.
        mode_type : str
            Mode type ("feature_importance" or "manual")
        mode_name : str
            Mode name (analysis name or feature selector name)
        n_top : int
            Number of top features (used in automatic filename)
        file_format : str
            File format extension (png, pdf, svg, etc.)
        dpi : int
            Resolution in dots per inch
        prefix : str
            Filename prefix ("violin" or "density")

        Returns
        -------
        None
            Saves figure to file and prints confirmation message

        Examples
        --------
        >>> # Automatic filename generation
        >>> plotter._save_figure(
        ...     fig, None, "feature_importance", "tree_analysis",
        ...     10, "png", 300, "violin"
        ... )
        Figure saved to: violin_tree_analysis_top10.png

        >>> # Custom filename
        >>> plotter._save_figure(
        ...     fig, "my_plot", "manual", "selector",
        ...     5, "pdf", 300, "density"
        ... )
        Figure saved to: my_plot.pdf
        """
        if filename is None:
            if mode_type == "feature_importance":
                filename = f"{prefix}_{mode_name}_top{n_top}.{file_format}"
            else:
                filename = f"{prefix}_{mode_name}.{file_format}"

        if not filename.endswith(f".{file_format}"):
            filename = f"{filename}.{file_format}"

        fig.savefig(filename, dpi=dpi, bbox_inches="tight")
        print(f"Figure saved to: {filename}")
