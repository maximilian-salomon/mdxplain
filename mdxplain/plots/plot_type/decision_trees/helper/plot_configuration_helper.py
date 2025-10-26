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
Plot configuration helper for decision tree plotting.

Provides static methods for validation, configuration, and dimension calculation.
All methods are stateless and receive required state as parameters.
"""

from ....helper.grid_layout_helper import GridLayoutHelper
from ....helper.validation_helper import ValidationHelper
from .decision_tree_visualizer import DecisionTreeVisualizer
from .decision_tree_visualization_config import DecisionTreeVisualizationConfig
from .tree_position_calculator import TreePositionCalculator

TREE_CONFIG = DecisionTreeVisualizationConfig()


class PlotConfigurationHelper:
    """
    Stateless helper for plot configuration and validation.

    All methods are static and receive required state as parameters.
    Code is copied 1:1 from DecisionTreePlotter for consistency.

    Examples
    --------
    >>> fi_data, metadata = PlotConfigurationHelper.validate_and_get_data(
    ...     pipeline_data, "tree_analysis"
    ... )
    """

    @staticmethod
    def validate_and_get_data(pipeline_data, feature_importance_name):
        """
        Validate and extract feature importance data.

        Parameters
        ----------
        pipeline_data : PipelineData
            Pipeline data container
        feature_importance_name : str
            Name of feature importance analysis

        Returns
        -------
        tuple
            (fi_data, feature_metadata)

        Raises
        ------
        ValueError
            If analyzer type is not decision_tree

        Examples
        --------
        >>> fi_data, metadata = PlotConfigurationHelper.validate_and_get_data(
        ...     pipeline_data, "analysis"
        ... )
        """
        ValidationHelper.validate_feature_importance_exists(
            pipeline_data, feature_importance_name
        )

        # Get feature importance data
        fi_data = pipeline_data.feature_importance_data[feature_importance_name]

        # Validate analyzer type
        if fi_data.analyzer_type != "decision_tree":
            raise ValueError(
                f"decision_trees() only works with decision_tree analyzer, "
                f"got '{fi_data.analyzer_type}'"
            )

        selector_name = fi_data.feature_selector
        feature_metadata = pipeline_data.get_selected_metadata(selector_name)
        return fi_data, feature_metadata

    @staticmethod
    def calculate_effective_depth(fi_data, max_depth_display):
        """
        Calculate effective tree depth for sizing.

        Checks all trees in feature importance data to find maximum depth.
        Uses config default if no models found.

        Parameters
        ----------
        fi_data : FeatureImportanceData
            Feature importance data
        max_depth_display : int or None
            User-specified max depth

        Returns
        -------
        int
            Effective depth for visualization

        Examples
        --------
        >>> depth = PlotConfigurationHelper.calculate_effective_depth(
        ...     fi_data, None
        ... )
        >>> print(depth)
        4
        """
        if max_depth_display is None:
            max_depth = TREE_CONFIG.default_depth_fallback
            for metadata in fi_data.metadata:
                model = metadata.get("model")
                if model:
                    max_depth = max(max_depth, model.tree_.max_depth)
            return max_depth
        return max_depth_display

    @staticmethod
    def calculate_subplot_sizes(fi_data, feature_metadata,
                                  max_depth_display, effective_depth,
                                  subplot_width, subplot_height):
        """
        Calculate dynamic subplot sizes based on tree width and depth.

        Parameters
        ----------
        fi_data : FeatureImportanceData
            Feature importance data
        feature_metadata : list
            Feature metadata
        max_depth_display : int or None
            Max depth
        effective_depth : int
            Effective tree depth
        subplot_width : float
            Initial subplot width
        subplot_height : float
            Initial subplot height

        Returns
        -------
        tuple
            (subplot_width, subplot_height)

        Examples
        --------
        >>> width, height = PlotConfigurationHelper.calculate_subplot_sizes(
        ...     fi_data, metadata, None, 4, 10.0, 8.0
        ... )
        """
        max_tree_width = PlotConfigurationHelper.calculate_required_tree_widths(
            fi_data, feature_metadata, max_depth_display
        )

        if subplot_width == TREE_CONFIG.default_subplot_width:
            subplot_width = max(max_tree_width / TREE_CONFIG.width_divisor, TREE_CONFIG.min_subplot_width)

        if subplot_height == TREE_CONFIG.default_subplot_height:
            # Estimate y_spacing (Just played with values to get reasonable heights)
            estimated_y_spacing = max(
                TREE_CONFIG.base_y_spacing,
                TREE_CONFIG.y_spacing_formula_base - (effective_depth - TREE_CONFIG.y_spacing_depth_offset) * TREE_CONFIG.y_spacing_depth_factor
            )
            # Height: depth levels * spacing + margins, converted to inches
            subplot_height = max(
                (effective_depth * estimated_y_spacing + TREE_CONFIG.height_margin) / TREE_CONFIG.height_divisor,
                TREE_CONFIG.min_subplot_height
            )

        return subplot_width, subplot_height

    @staticmethod
    def calculate_grid_layout(fi_data, max_cols):
        """
        Calculate grid layout dimensions.

        Parameters
        ----------
        fi_data : FeatureImportanceData
            Feature importance data
        max_cols : int
            Maximum columns

        Returns
        -------
        tuple
            (n_comparisons, n_rows, n_cols, hspace_dynamic)

        Examples
        --------
        >>> n, rows, cols, hspace = PlotConfigurationHelper.calculate_grid_layout(
        ...     fi_data, 2
        ... )
        """
        n_comparisons = len(fi_data.data)
        _, n_rows, n_cols = GridLayoutHelper.compute_uniform_grid_layout(
            n_comparisons, max_cols
        )
        hspace_dynamic = TREE_CONFIG.grid_hspace_dynamic
        return n_comparisons, n_rows, n_cols, hspace_dynamic

    @staticmethod
    def determine_separate_trees_mode(separate_trees, effective_depth):
        """
        Determine whether to use separate trees mode.

        Parameters
        ----------
        separate_trees : bool or str
            User preference (True, False, or "auto")
        effective_depth : int
            Tree depth

        Returns
        -------
        bool
            True for separate trees mode

        Raises
        ------
        ValueError
            If depth > 6

        Examples
        --------
        >>> use_sep = PlotConfigurationHelper.determine_separate_trees_mode(
        ...     "auto", 5
        ... )
        """
        PlotConfigurationHelper.validate_tree_depth(effective_depth)

        if separate_trees == "auto":
            return PlotConfigurationHelper.auto_determine_mode(effective_depth)

        return bool(separate_trees)

    @staticmethod
    def validate_tree_depth(effective_depth):
        """
        Validate tree depth is visualizable.

        Parameters
        ----------
        effective_depth : int
            Tree depth

        Raises
        ------
        ValueError
            If depth > max_visualizable_depth from config

        Examples
        --------
        >>> PlotConfigurationHelper.validate_tree_depth(4)
        """
        if effective_depth > TREE_CONFIG.max_visualizable_depth:
            raise ValueError(
                f"Tree depth={effective_depth} is too large for visualization!\n"
                f"Use 'max_depth_display={TREE_CONFIG.max_visualizable_depth}' (or smaller) to limit tree depth."
            )

    @staticmethod
    def auto_determine_mode(effective_depth):
        """
        Automatically determine mode based on depth.

        Parameters
        ----------
        effective_depth : int
            Tree depth

        Returns
        -------
        bool
            True for separate trees

        Examples
        --------
        >>> use_sep = PlotConfigurationHelper.auto_determine_mode(5)
        ℹ️  depth=5 ≥ 5: Automatically using separate_trees=True
        """
        use_separate = effective_depth >= TREE_CONFIG.auto_mode_separate_threshold
        mode_msg = "separate_trees=True" if use_separate else "Grid mode"
        threshold = str(TREE_CONFIG.auto_mode_separate_threshold) if use_separate else str(TREE_CONFIG.auto_mode_separate_threshold - 1)
        operator = "≥" if use_separate else "≤"
        print(f"ℹ️  depth={effective_depth} {operator} {threshold}: "
              f"Automatically using {mode_msg}")
        return use_separate

    @staticmethod
    def apply_dimension_limits(use_separate_trees, effective_depth,
                                 subplot_width, subplot_height,
                                 width_scale_factor, height_scale_factor, dpi):
        """
        Apply dimension limits and scale factors.

        Parameters
        ----------
        use_separate_trees : bool
            Whether using separate trees mode
        effective_depth : int
            Tree depth
        subplot_width : float
            Subplot width
        subplot_height : float
            Subplot height
        width_scale_factor : float
            Width multiplier
        height_scale_factor : float
            Height multiplier
        dpi : int
            DPI setting

        Returns
        -------
        tuple
            (subplot_width, subplot_height, dpi)

        Examples
        --------
        >>> w, h, d = PlotConfigurationHelper.apply_dimension_limits(
        ...     True, 5, 12.0, 10.0, 1.0, 1.0, 300
        ... )
        """
        subplot_width, subplot_height = PlotConfigurationHelper.set_depth_based_dimensions(
            effective_depth, subplot_width, subplot_height
        )
        subplot_width *= width_scale_factor
        subplot_height *= height_scale_factor

        if use_separate_trees:
            dpi = PlotConfigurationHelper.adjust_dpi_if_needed(subplot_width, subplot_height, dpi)

        return subplot_width, subplot_height, dpi

    @staticmethod
    def set_depth_based_dimensions(effective_depth, subplot_width, subplot_height):
        """
        Set dimensions based on tree depth.

        Parameters
        ----------
        effective_depth : int
            Tree depth
        subplot_width : float
            Current width
        subplot_height : float
            Current height

        Returns
        -------
        tuple
            (subplot_width, subplot_height)

        Examples
        --------
        >>> w, h = PlotConfigurationHelper.set_depth_based_dimensions(
        ...     5, 15.0, 12.0
        ... )
        """
        width_limit = TREE_CONFIG.depth_width_limits.get(
            effective_depth, TREE_CONFIG.default_width_limit
        )
        height_limit = TREE_CONFIG.depth_height_limits.get(
            effective_depth, TREE_CONFIG.default_height_limit
        )
        return min(subplot_width, width_limit), min(subplot_height, height_limit)

    @staticmethod
    def adjust_dpi_if_needed(subplot_width, subplot_height, dpi):
        """
        Reduce DPI if dimensions are too large.

        Parameters
        ----------
        subplot_width : float
            Width in inches
        subplot_height : float
            Height in inches
        dpi : int
            Current DPI

        Returns
        -------
        int
            Adjusted DPI

        Examples
        --------
        >>> new_dpi = PlotConfigurationHelper.adjust_dpi_if_needed(
        ...     25.0, 20.0, 300
        ... )
        """
        max_dimension = max(subplot_width, subplot_height)
        if max_dimension > TREE_CONFIG.max_dimension_for_dpi_reduction:
            print(f"Large tree dimensions ({max_dimension:.1f}\"): DPI {dpi} → {TREE_CONFIG.reduced_dpi}")
            return TREE_CONFIG.reduced_dpi
        return dpi

    @staticmethod
    def validate_output_methods(render, save_fig):
        """
        Validate that at least one output method is enabled.

        Parameters
        ----------
        render : bool
            Whether to render
        save_fig : bool
            Whether to save

        Raises
        ------
        ValueError
            If both are False

        Examples
        --------
        >>> PlotConfigurationHelper.validate_output_methods(True, False)
        """
        if not render and not save_fig:
            raise ValueError(
                "render=False and save_fig=False: No output method!\n"
                "At least one must be True."
            )

    @staticmethod
    def calculate_required_tree_widths(
        fi_data,
        feature_metadata,
        max_depth_display
    ):
        """
        Pre-analyze all trees to determine required widths.

        Creates visualizers for all trees and computes their layout
        to determine the maximum width needed for proper display.

        Parameters
        ----------
        fi_data : FeatureImportanceData
            Feature importance data containing models
        feature_metadata : list
            Feature metadata for creating feature names
        max_depth_display : int, optional
            Maximum depth to display

        Returns
        -------
        float
            Maximum width needed across all trees in tree coordinates

        Examples
        --------
        >>> max_width = PlotConfigurationHelper.calculate_required_tree_widths(
        ...     fi_data, feature_metadata, max_depth_display=3
        ... )
        >>> print(f"Max tree width: {max_width:.1f}")
        Max tree width: 2500.0
        """
        widths = []
        for metadata in fi_data.metadata:
            width = PlotConfigurationHelper.calculate_single_tree_width(
                metadata, feature_metadata, max_depth_display
            )
            if width is not None:
                widths.append(width)

        return max(widths) if widths else TREE_CONFIG.default_tree_width_fallback

    @staticmethod
    def calculate_single_tree_width(metadata, feature_metadata, max_depth_display):
        """
        Calculate width for single tree.

        Parameters
        ----------
        metadata : dict
            Tree metadata
        feature_metadata : list
            Feature metadata
        max_depth_display : int or None
            Max depth

        Returns
        -------
        float or None
            Tree width or None if no model

        Examples
        --------
        >>> width = PlotConfigurationHelper.calculate_single_tree_width(
        ...     metadata, feature_metadata, 3
        ... )
        """
        model = metadata.get("model")
        if model is None:
            return None

        class_names = PlotConfigurationHelper.extract_class_names(metadata)
        visualizer = DecisionTreeVisualizer(
            model, feature_metadata, class_names, max_depth_display
        )
        visualizer.build_graph()
        TreePositionCalculator.compute_positions(
            visualizer.tree_, visualizer.node_sizes, visualizer.positions, visualizer.max_depth
        )

        return PlotConfigurationHelper.extract_width_from_visualizer(visualizer)

    @staticmethod
    def extract_width_from_visualizer(visualizer):
        """
        Extract width from visualizer positions.

        Parameters
        ----------
        visualizer : DecisionTreeVisualizer
            Visualizer with computed positions

        Returns
        -------
        float or None
            Width in tree coordinates

        Examples
        --------
        >>> width = PlotConfigurationHelper.extract_width_from_visualizer(
        ...     visualizer
        ... )
        """
        if not visualizer.positions:
            return None

        all_x = [visualizer.positions[node]['x'] for node in visualizer.positions]
        return max(all_x) - min(all_x) + TREE_CONFIG.width_extra_margin

    @staticmethod
    def extract_class_names(metadata):
        """
        Extract class names from sub-comparison metadata.

        Parameters
        ----------
        metadata : dict
            Sub-comparison metadata containing class_names or sub_comparison_name

        Returns
        -------
        List[str]
            List of two class names for binary classification

        Examples
        --------
        >>> names = PlotConfigurationHelper.extract_class_names(
        ...     {"class_names": ["A", "B"]}
        ... )
        >>> print(names)
        ['A', 'B']

        >>> names = PlotConfigurationHelper.extract_class_names(
        ...     {"sub_comparison_name": "A_vs_B"}
        ... )
        >>> print(names)
        ['A', 'B']
        """
        # Get from metadata if available
        if "class_names" in metadata:
            return metadata["class_names"]

        # Fallback: infer from sub_comparison_name
        sub_comp_name = metadata.get("sub_comparison_name", "")
        if "_vs_" in sub_comp_name:
            parts = sub_comp_name.split("_vs_")
            return [parts[0], parts[1]]

        # Final fallback
        return ["Class 0", "Class 1"]
