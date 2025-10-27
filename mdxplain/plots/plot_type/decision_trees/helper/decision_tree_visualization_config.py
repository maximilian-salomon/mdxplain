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
Configuration for decision tree visualization.

Centralizes all hardcoded values for tree plotting and visualization.
"""

from dataclasses import dataclass, field
from typing import Dict


@dataclass
class DecisionTreeVisualizationConfig:
    """
    Configuration for decision tree visualization.

    Centralizes all hardcoded values for dimensions, spacing, fonts,
    and other visualization parameters.

    Examples
    --------
    >>> config = DecisionTreeVisualizationConfig()
    >>> print(config.max_visualizable_depth)
    6
    >>> print(config.default_depth_fallback)
    3
    """

    # Tree depth limits
    # Maximum tree depth that can be visualized
    max_visualizable_depth: int = 6
    # Default depth when no model is found (changed from 5 to 3)
    default_depth_fallback: int = 3

    # Spacing calculations
    # Minimum vertical spacing between tree levels
    base_y_spacing: int = 120
    # Base value for y_spacing formula
    y_spacing_formula_base: int = 450
    # Factor multiplied by depth in y_spacing formula
    y_spacing_depth_factor: int = 18
    # Offset subtracted from depth in y_spacing formula
    y_spacing_depth_offset: int = 3

    # Height calculations
    # Default subplot height compared against user input
    default_subplot_height: float = 8.0
    # Margin added to calculated height in pixels
    height_margin: int = 400
    # Divisor to convert pixel height to inches
    height_divisor: int = 100
    # Minimum allowed subplot height in inches
    min_subplot_height: float = 10.0

    # Width calculations
    # Minimum allowed subplot width in inches
    min_subplot_width: float = 10.0
    # Default subplot width compared against user input
    default_subplot_width: float = 10.0
    # Divisor for converting tree width to subplot width
    width_divisor: float = 160.0
    # Extra margin added to tree width calculation
    width_extra_margin: int = 200
    # Fallback width when no tree widths can be calculated
    default_tree_width_fallback: float = 1000.0

    # Dimension limits per depth
    # Maximum width limits for different tree depths
    depth_width_limits: Dict[int, float] = field(default_factory=lambda: {
        6: 100.0,
        5: 65.0,
        4: 50.0,
    })
    # Maximum height limits for different tree depths
    depth_height_limits: Dict[int, float] = field(default_factory=lambda: {
        6: 35.0,
        5: 30.0,
        4: 30.0,
    })
    # Default width limit for depths not in depth_width_limits
    default_width_limit: float = 40.0
    # Default height limit for depths not in depth_height_limits
    default_height_limit: float = 30.0

    # DPI settings
    # Maximum dimension (width or height) before DPI reduction
    max_dimension_for_dpi_reduction: float = 60.0
    # Reduced DPI value for large figures
    reduced_dpi: int = 150

    # GridSpec settings
    # Horizontal spacing between grid subplots
    grid_wspace: float = 0.02
    # Vertical spacing between grid rows
    grid_hspace_dynamic: float = 0.15
    # Top margin of grid layout
    grid_top: float = 0.96
    # Bottom margin of grid layout
    grid_bottom: float = 0.02
    # Left margin of grid layout
    grid_left: float = 0.02
    # Right margin of grid layout
    grid_right: float = 0.98

    # Visual styling
    # Border line width around subplots
    subplot_border_width: float = 10.0
    # Padding between subplot and its title
    subplot_title_pad: float = 15.0
    # Y-position of main title
    main_title_y: float = 0.98

    # Font sizes per depth (subplot titles)
    # Font sizes for subplot titles at different depths
    subplot_title_fontsize: Dict[int, int] = field(default_factory=lambda: {
        6: 42,
        5: 38,
        4: 28,
    })
    # Default font size for subplot titles at unlisted depths
    default_subplot_title_fontsize: int = 14

    # Font sizes per depth (main title)
    # Font sizes for main figure title at different depths
    main_title_fontsize: Dict[int, int] = field(default_factory=lambda: {
        6: 54,
        5: 50,
        4: 36,
    })
    # Default font size for main title at unlisted depths
    default_main_title_fontsize: int = 18

    # Separate tree title settings
    # Font size for separate tree mode titles
    separate_tree_title_fontsize: int = 14
    # Padding between tree and title in separate mode
    separate_tree_title_pad: float = 20.0

    # Visualizer-specific settings
    # Font size for depth 6 trees (64 leaf nodes)
    visualizer_fontsize_depth_6: float = 6.5
    # Font size for depth 5 trees (32 leaf nodes)
    visualizer_fontsize_depth_5: float = 7.0
    # Minimum allowed font size
    visualizer_fontsize_min: float = 6.0
    # Base font size for depth calculation
    visualizer_fontsize_base: float = 10.0
    # Factor for depth-based font size reduction
    visualizer_fontsize_factor: float = 0.3

    # Offset subtracted from node font size for edge labels
    edge_fontsize_offset: float = 1.0
    # Minimum font size for edge labels
    edge_fontsize_min: float = 5.0

    # BBox padding for deep trees (depth >= 5)
    node_pad_deep: float = 0.2
    # BBox padding for shallow trees (depth < 5)
    node_pad_shallow: float = 0.3
    # Depth threshold between deep and shallow padding
    node_pad_depth_threshold: int = 5

    # Node spacing
    # Node spacing for depth 6 trees
    default_spacing_depth_6: float = 80.0
    # Node spacing for depth 5 trees
    default_spacing_depth_5: float = 100.0
    # Fallback node spacing for other depths
    default_spacing_fallback: float = 150.0

    # Minimum optimal spacing between nodes
    optimal_spacing_min: float = 30.0
    # Maximum optimal spacing between nodes
    optimal_spacing_max: float = 300.0
    # Spacing for single-node trees
    optimal_spacing_single_node: float = 150.0

    # Axis padding
    # Base percentage for x-axis padding
    padding_x_base: float = 0.12
    # Factor for depth-based x-padding reduction
    padding_x_depth_factor: float = 0.01
    # Minimum x-padding percentage
    padding_x_min_percent: float = 0.02
    # Minimum absolute x-padding in pixels
    padding_x_min_absolute: float = 80.0

    # Y-axis padding as percentage of range
    padding_y_percent: float = 0.35
    # Minimum absolute y-padding in pixels
    padding_y_min_absolute: float = 150.0
    # Default y-range when no nodes exist
    padding_y_default_range: float = 150.0
    # Default x-range when no nodes exist
    padding_default_x_range: float = 250.0

    # Text wrapping
    # Maximum line length for wrapped discrete labels
    discrete_label_wrap_length: int = 40

    # Depth thresholds for auto mode
    # Depth threshold for automatic separate trees mode
    auto_mode_separate_threshold: int = 5
