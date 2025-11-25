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
Decision tree visualizer for feature importance analysis.

Provides visualization capabilities for sklearn DecisionTreeClassifier models
with support for multi-class classification and feature-type-specific labeling.
"""

import matplotlib.pyplot as plt
from sklearn.tree import _tree
from typing import Dict, List, Optional

from .....utils.feature_metadata_utils import FeatureMetadataUtils
from .decision_tree_visualization_config import DecisionTreeVisualizationConfig
from .edge_label_formatter import EdgeLabelFormatter
from .feature_label_builder import FeatureLabelBuilder
from .node_text_formatter import NodeTextFormatter
from .tree_position_calculator import TreePositionCalculator

# Global config instance
TREE_CONFIG = DecisionTreeVisualizationConfig()


class DecisionTreeVisualizer:
    """
    Visualize decision trees with feature-type-specific formatting.

    Creates visual representations of trained DecisionTreeClassifier models
    with support for multi-class classification, feature-type-specific edge
    labels, and depth limiting.

    Examples
    --------
    >>> visualizer = DecisionTreeVisualizer(
    ...     clf, feature_names, class_names, type_metadata
    ... )
    >>> fig = visualizer.visualize()
    """

    def __init__(
        self,
        clf,
        feature_metadata: List[Dict],
        class_names: List[str],
        max_depth_display: Optional[int] = None,
        short_labels: Optional[bool] = None,
        short_naming: Optional[bool] = None,
        short_layout: bool = False,
        short_edge_labels: Optional[bool] = None,
        wrap_length: int = 40,
        hide_node_frames: Optional[bool] = None,
        show_edge_symbols: Optional[bool] = None,
        hide_feature_type_prefix: Optional[bool] = None,
        hide_path: Optional[bool] = None,
        edge_symbol_fontsize: Optional[int] = None
    ):
        """
        Initialize decision tree visualizer.

        Parameters
        ----------
        clf : DecisionTreeClassifier
            Trained scikit-learn decision tree classifier
        feature_metadata : List[Dict]
            Complete feature metadata array for all features
        class_names : List[str]
            List of class names for counting
        max_depth_display : int, optional
            Maximum depth to display (None for full tree)
        short_labels : bool, optional, default=None
            Use short discrete labels (NC vs Non-Contact).
            If None, determined by short_layout.
        short_naming : bool, optional, default=None
            Truncate class names to 16 chars with [...] pattern.
            If None, determined by short_layout.
        short_layout : bool, default=False
            Minimal tree layout + enables all short options (if not explicitly set)
        short_edge_labels : bool, optional, default=None
            Show only values/conditions on edges (e.g., 'Contact' or '≤ 3.50 Å').
            If None, determined by short_layout.
        wrap_length : int, default=40
            Maximum line length for text wrapping in node labels
        hide_node_frames : bool, optional, default=None
            Hide frame counts in non-root nodes, showing only percentages.
            If None, determined by short_layout.
        show_edge_symbols : bool, optional, default=None
            Show only symbols on edges (✓ for left/true, ✗ for right/false).
            If None, determined by short_layout.
        hide_feature_type_prefix : bool, optional, default=None
            Hide feature type prefix in labels.
            If None, determined by short_layout.
        hide_path : bool, optional, default=None
            Hide decision path in non-root nodes.
            If None, set to True when short_layout=True.
        edge_symbol_fontsize : int, optional
            Font size for edge symbols. If None, uses edge_fontsize

        Returns
        -------
        None
            Initializes DecisionTreeVisualizer instance
        """
        self.clf = clf
        self.tree_ = clf.tree_
        self.feature_metadata = feature_metadata
        self.class_names = class_names
        self.max_depth_display = max_depth_display
        self.wrap_length = wrap_length
        self.hide_node_frames = hide_node_frames
        self.show_edge_symbols = show_edge_symbols
        self.hide_feature_type_prefix = hide_feature_type_prefix
        self.edge_symbol_fontsize = edge_symbol_fontsize

        self._apply_short_options(short_labels, short_naming, short_layout, short_edge_labels, hide_path)
        self._initialize_tree_metrics()
        self._calculate_font_sizes()

    def _apply_short_options(self, short_labels, short_naming, short_layout, short_edge_labels, hide_path):
        """
        Apply short label options and configure class names.

        Parameters
        ----------
        short_labels : bool or None
            Use short discrete labels (None = use short_layout)
        short_naming : bool or None
            Truncate class names (None = use short_layout)
        short_layout : bool
            Enable minimal layout (enables all options if they are None)
        short_edge_labels : bool or None
            Show only values on edges (None = use short_layout)
        hide_path : bool or None
            Hide decision path (None = use short_layout)

        Returns
        -------
        None
            Sets short option flags and modifies self.class_names
        """
        self.short_layout = short_layout
        self.short_labels = self._resolve_short_option(short_labels, short_layout)
        self.short_naming = self._resolve_short_option(short_naming, short_layout)
        self.short_edge_labels = self._resolve_short_option(short_edge_labels, short_layout)
        self.hide_path = self._resolve_short_option(hide_path, short_layout)

        # Zusätzliche Parameter automatisch aktivieren bei short_layout (nur wenn None)
        self.show_edge_symbols = self._resolve_short_option(self.show_edge_symbols, short_layout)
        self.hide_feature_type_prefix = self._resolve_short_option(self.hide_feature_type_prefix, short_layout)
        self.hide_node_frames = self._resolve_short_option(self.hide_node_frames, short_layout)

        if self.short_naming:
            self.class_names = [
                NodeTextFormatter.shorten_class_name(name) for name in self.class_names
            ]

    def _resolve_short_option(self, option, short_layout):
        """
        Resolve short option value based on layout setting.

        Only applies short_layout if option is None (not explicitly set by user).
        If user explicitly sets option to False, it stays False.

        Parameters
        ----------
        option : bool or None
            Individual short option flag (None = not set by user)
        short_layout : bool
            Master short layout flag

        Returns
        -------
        bool
            User's explicit choice if set, otherwise short_layout value

        Examples
        --------
        >>> # User didn't set parameter (None), use short_layout
        >>> self._resolve_short_option(None, True)
        True

        >>> # User explicitly set to False, respect that
        >>> self._resolve_short_option(False, True)
        False

        >>> # User explicitly set to True, respect that
        >>> self._resolve_short_option(True, False)
        True
        """
        if option is None:
            return short_layout
        return option

    def _initialize_tree_metrics(self):
        """
        Initialize tree data structures and calculate total counts.

        Returns
        -------
        None
            Initializes labels, positions, counts, and max_depth
        """
        self.labels = {}
        self.node_sizes = {}
        self.positions = {}
        self.y_positions = {}
        self.max_depth = min(
            self.tree_.max_depth,
            self.max_depth_display if self.max_depth_display else self.tree_.max_depth
        )

        # Calculate total counts per class
        root_value = self.tree_.value[0][0] * self.tree_.n_node_samples[0]
        self.total_counts = [int(root_value[i]) for i in range(len(self.class_names))]

        # Track node with maximum score for highlighting
        self.max_score_node = None
        self.max_score = float('-inf')

    def _calculate_font_sizes(self):
        """
        Calculate font sizes and spacing based on tree depth.

        Returns
        -------
        None
            Sets y_spacing, fontsize, edge_fontsize, and node_pad
        """
        self.y_spacing = max(
            TREE_CONFIG.base_y_spacing,
            TREE_CONFIG.y_spacing_formula_base - (self.max_depth - TREE_CONFIG.y_spacing_depth_offset) * TREE_CONFIG.y_spacing_depth_factor
        )

        # Font size: smaller for deep trees to create narrower nodes
        if self.max_depth == 6:
            self.fontsize = TREE_CONFIG.visualizer_fontsize_depth_6
        elif self.max_depth == 5:
            self.fontsize = TREE_CONFIG.visualizer_fontsize_depth_5
        else:
            self.fontsize = max(
                TREE_CONFIG.visualizer_fontsize_min,
                TREE_CONFIG.visualizer_fontsize_base - self.max_depth * TREE_CONFIG.visualizer_fontsize_factor
            )

        self.edge_fontsize = max(
            TREE_CONFIG.edge_fontsize_min,
            self.fontsize - TREE_CONFIG.edge_fontsize_offset
        )

        # BBox padding: smaller for deep trees to create narrower nodes
        self.node_pad = TREE_CONFIG.node_pad_deep if self.max_depth >= TREE_CONFIG.node_pad_depth_threshold else TREE_CONFIG.node_pad_shallow

    def build_graph(self):
        """
        Build graph by traversing tree and collecting node labels.

        Recursively traverses the tree to build node labels, sizes,
        and identifies the node with maximum score for highlighting.

        Returns
        -------
        None
            Populates self.labels and self.node_sizes
        """
        self._recurse_build_graph(0, 0)

    def _recurse_build_graph(self, node, depth, path=None):
        """
        Recursively build graph structure for decision tree node.

        Traverses tree to create node labels, compute sizes, and identify
        the node with maximum score for highlighting.

        Parameters
        ----------
        node : int
            Current node index in tree
        depth : int
            Current depth level in tree
        path : list, optional
            List of edge labels from root to current node

        Returns
        -------
        None
            Populates self.labels, self.node_sizes, self.max_score_node
        """
        if path is None:
            path = []

        if self.max_depth_display and depth > self.max_depth_display:
            return

        counts, percentages = self._calculate_node_statistics(node)
        self._update_max_score(node, percentages)

        if self.tree_.feature[node] != _tree.TREE_UNDEFINED:
            self._process_decision_node(node, depth, path, counts, percentages)
        else:
            self._process_leaf_node(node, path, counts, percentages)

    def _calculate_node_statistics(self, node):
        """
        Calculate class counts and percentages for node.

        Parameters
        ----------
        node : int
            Node index in tree

        Returns
        -------
        tuple
            (counts, percentages) where counts is list of int and
            percentages is list of float
        """
        value = self.tree_.value[node][0] * self.tree_.n_node_samples[node]
        counts = [int(value[i]) for i in range(len(self.class_names))]
        percentages = [
            (count / self.total_counts[i]) * 100 if self.total_counts[i] > 0 else 0
            for i, count in enumerate(counts)
        ]
        return counts, percentages

    def _update_max_score(self, node, percentages):
        """
        Update maximum score tracking for node highlighting.

        Parameters
        ----------
        node : int
            Node index in tree
        percentages : list of float
            Class percentages for this node

        Returns
        -------
        None
            Updates self.max_score and self.max_score_node
        """
        score = percentages[0] - sum(percentages[1:])
        if score > self.max_score:
            self.max_score = score
            self.max_score_node = node

    def _create_node_text_and_size(self, node, feature_idx, threshold, path, counts, percentages):
        """
        Create node text and calculate size.

        Parameters
        ----------
        node : int
            Node index
        feature_idx : int
            Feature index for split
        threshold : float
            Decision threshold
        path : list of str
            Decision path
        counts : list of int
            Class counts
        percentages : list of float
            Class percentages

        Returns
        -------
        None
            Updates self.labels and self.node_sizes
        """
        feature_type = FeatureMetadataUtils.get_feature_type(
            self.feature_metadata, feature_idx
        )
        feature_name = FeatureMetadataUtils.get_feature_name(
            self.feature_metadata, feature_idx
        )
        discrete_labels = NodeTextFormatter.get_discrete_labels(
            feature_idx, self.feature_metadata, self.short_labels
        )

        node_text = NodeTextFormatter.format_node_text(
            path, counts, percentages, feature_type, feature_name,
            threshold, discrete_labels, self.class_names, self.total_counts,
            self.hide_path, self.wrap_length, self.hide_node_frames,
            self.hide_feature_type_prefix, self.feature_metadata, feature_idx
        )
        self.labels[node] = node_text

        width, height = TreePositionCalculator.get_text_size(
            node_text, self.fontsize, self.node_pad
        )
        self.node_sizes[node] = (width, height)

    def _create_path_labels(self, feature_idx, threshold):
        """
        Create path labels for decision edges.

        Parameters
        ----------
        feature_idx : int
            Feature index
        threshold : float
            Decision threshold

        Returns
        -------
        tuple of str
            (left_label, right_label)
        """
        return EdgeLabelFormatter.get_path_labels(
            feature_idx, threshold, self.feature_metadata,
            self.short_labels, self.hide_feature_type_prefix,
            self.show_edge_symbols, NodeTextFormatter.get_discrete_labels
        )

    def _process_decision_node(self, node, depth, path, counts, percentages):
        """
        Process decision node by creating label and recursing to children.

        Parameters
        ----------
        node : int
            Node index in tree
        depth : int
            Current depth in tree
        path : list of str
            Decision path to this node
        counts : list of int
            Class counts for this node
        percentages : list of float
            Class percentages for this node

        Returns
        -------
        None
            Updates self.labels and self.node_sizes, recurses to children
        """
        feature_idx = self.tree_.feature[node]
        threshold = self.tree_.threshold[node]

        self._create_node_text_and_size(node, feature_idx, threshold, path, counts, percentages)
        left_label, right_label = self._create_path_labels(feature_idx, threshold)
        self._recurse_to_children(node, depth, path, left_label, right_label)

    def _recurse_to_children(self, node, depth, path, left_label, right_label):
        """
        Recursively process child nodes if within depth limit.

        Parameters
        ----------
        node : int
            Current node index
        depth : int
            Current depth
        path : list of str
            Decision path
        left_label : str
            Left edge label
        right_label : str
            Right edge label

        Returns
        -------
        None
            Calls _recurse_build_graph on children
        """
        if not self.max_depth_display or depth < self.max_depth_display:
            left_child = self.tree_.children_left[node]
            right_child = self.tree_.children_right[node]
            self._recurse_build_graph(left_child, depth + 1, path + [left_label])
            self._recurse_build_graph(right_child, depth + 1, path + [right_label])

    def _process_leaf_node(self, node, path, counts, percentages):
        """
        Process leaf node by creating label.

        Parameters
        ----------
        node : int
            Node index in tree
        path : list of str
            Decision path to this leaf
        counts : list of int
            Class counts for this leaf
        percentages : list of float
            Class percentages for this leaf

        Returns
        -------
        None
            Updates self.labels and self.node_sizes
        """
        node_text = NodeTextFormatter.format_leaf_node_text(path, counts, percentages, self.class_names, self.total_counts, self.hide_path, self.wrap_length, self.hide_node_frames)
        self.labels[node] = node_text
        width, height = TreePositionCalculator.get_text_size(node_text, self.fontsize, self.node_pad)
        self.node_sizes[node] = (width, height)

    def plot_tree(self, ax):
        """
        Plot tree on given axes.

        Recursively draws all nodes and edges, then adjusts axis limits
        and formatting for clean presentation.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            Axes object to plot on

        Returns
        -------
        None
            Modifies ax in place

        Examples
        --------
        >>> fig, ax = plt.subplots(figsize=(12, 8))
        >>> visualizer.plot_tree(ax)
        >>> plt.savefig('decision_tree.png')
        """
        all_x = [self.positions[node]['x'] for node in self.positions]
        all_y = [-depth * self.y_spacing for depth in range(self.max_depth + 1)]

        # Plot nodes
        self._plot_node(ax, 0, 0)

        # Adjust limits with proportional padding
        x_range = max(all_x) - min(all_x) if all_x else TREE_CONFIG.padding_default_x_range
        y_range = abs(min(all_y)) if all_y else TREE_CONFIG.padding_y_default_range

        # Shallow trees (wider) need more x-padding to prevent exceeding bounds
        padding_x_percent = max(
            TREE_CONFIG.padding_x_min_percent,
            TREE_CONFIG.padding_x_base - self.max_depth * TREE_CONFIG.padding_x_depth_factor
        )
        padding_x = max(x_range * padding_x_percent, TREE_CONFIG.padding_x_min_absolute)
        padding_y = max(y_range * TREE_CONFIG.padding_y_percent, TREE_CONFIG.padding_y_min_absolute)

        ax.set_xlim(min(all_x) - padding_x, max(all_x) + padding_x)
        ax.set_ylim(min(all_y) - padding_y, padding_y)

        # Hide ticks and labels but keep spines for border
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel('')
        ax.set_ylabel('')

    def _plot_node(self, ax, node, depth):
        """
        Recursively plot node and connections.

        Draws node text box, then recursively draws children and connecting edges
        with labeled conditions.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            Axes object to plot on
        node : int
            Current node index in tree
        depth : int
            Current depth level in tree

        Returns
        -------
        None
            Modifies ax in place by adding text and lines

        Examples
        --------
        >>> fig, ax = plt.subplots()
        >>> visualizer._plot_node(ax, 0, 0)  # Plot starting from root
        """
        x = self.positions[node]['x']
        y = -depth * self.y_spacing
        self.y_positions[node] = y

        node_text = self.labels[node]
        is_highlighted = node == self.max_score_node
        edgecolor = 'green' if is_highlighted else 'black'
        linewidth = 3 if is_highlighted else 1

        ax.text(x, y, node_text, ha='center', va='center', fontsize=self.fontsize,
                bbox=dict(boxstyle=f'round,pad={self.node_pad}', edgecolor=edgecolor,
                          facecolor='lightgrey', linewidth=linewidth))

        if self.tree_.feature[node] != _tree.TREE_UNDEFINED:
            left_child = self.tree_.children_left[node]
            right_child = self.tree_.children_right[node]

            # Only plot children if they exist in positions
            if left_child in self.positions and right_child in self.positions:
                self._plot_node(ax, left_child, depth + 1)
                self._plot_node(ax, right_child, depth + 1)

                # Draw edges and labels
                self._draw_edges_with_labels(ax, node, x, y, left_child, right_child)

    def _draw_edges_with_labels(self, ax, node, x, y, left_child, right_child):
        """
        Draw edges and labels to child nodes.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            Axes object to plot on
        node : int
            Parent node index
        x : float
            Parent node x position
        y : float
            Parent node y position
        left_child : int
            Left child node index
        right_child : int
            Right child node index

        Returns
        -------
        None
            Modifies ax in place

        Examples
        --------
        >>> visualizer._draw_edges_with_labels(ax, 0, 0.5, 0, 1, 2)
        """
        left_x = self.positions[left_child]['x']
        left_y = self.y_positions[left_child]
        right_x = self.positions[right_child]['x']
        right_y = self.y_positions[right_child]

        # Draw edges
        ax.plot([x, left_x], [y, left_y], 'k-')
        ax.plot([x, right_x], [y, right_y], 'k-')

        # Calculate midpoints
        left_mid_x = (x + left_x) / 2
        left_mid_y = (y + left_y) / 2
        right_mid_x = (x + right_x) / 2
        right_mid_y = (y + right_y) / 2

        # Get edge labels
        feature_idx = self.tree_.feature[node]
        threshold = self.tree_.threshold[node]
        left_label, right_label = EdgeLabelFormatter.get_edge_labels(
            feature_idx, threshold, self.feature_metadata, self.short_labels,
            self.short_edge_labels, self.wrap_length, self.show_edge_symbols,
            self.hide_feature_type_prefix
        )

        # Get edge colors based on reverse semantics
        left_color, right_color = self._get_edge_colors(feature_idx)

        # Get font size
        edge_label_fontsize = self._get_edge_label_fontsize()

        # Draw labels
        ax.text(left_mid_x, left_mid_y, left_label, ha='center', va='center',
                fontsize=edge_label_fontsize, color=left_color,
                bbox=dict(boxstyle='round,pad=0', facecolor='white', edgecolor='none'))
        ax.text(right_mid_x, right_mid_y, right_label, ha='center', va='center',
                fontsize=edge_label_fontsize, color=right_color,
                bbox=dict(boxstyle='round,pad=0', facecolor='white', edgecolor='none'))

    def _get_edge_colors(self, feature_idx):
        """
        Get edge colors based on reverse semantics setting.

        Parameters
        ----------
        feature_idx : int
            Feature index

        Returns
        -------
        tuple
            (left_color, right_color)

        Examples
        --------
        >>> colors = visualizer._get_edge_colors(5)
        >>> print(colors)  # ('green', 'red') or ('red', 'green')
        """
        feature_type = FeatureMetadataUtils.get_feature_type(
            self.feature_metadata, feature_idx
        )
        type_meta = FeatureMetadataUtils.get_top_level_metadata(
            feature_type, self.feature_metadata
        )

        reverse_edges = FeatureLabelBuilder.get_reverse_edge_semantics(type_meta)
        return ('red', 'green') if reverse_edges else ('green', 'red')

    def _get_edge_label_fontsize(self):
        """
        Get font size for edge labels.

        Returns
        -------
        int
            Font size for edge labels

        Examples
        --------
        >>> fontsize = visualizer._get_edge_label_fontsize()
        >>> print(fontsize)  # 10 or custom size
        """
        if self.show_edge_symbols:
            return self.edge_symbol_fontsize if self.edge_symbol_fontsize else self.edge_fontsize
        return self.edge_fontsize

    def visualize(self, ax=None, target_width=None):
        """
        Execute visualization steps.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            Axes to plot on (creates new figure if None)
        target_width : float, optional
            Target width in pixels for adaptive spacing.
            If provided, calculates optimal node spacing to fit width.

        Returns
        -------
        matplotlib.axes.Axes
            Axes with tree visualization
        """
        self.build_graph()

        # Calculate optimal spacing if target width provided
        if target_width is not None:
            optimal_spacing = TreePositionCalculator.calculate_optimal_spacing(self.tree_, self.node_sizes, target_width)
            TreePositionCalculator.compute_positions(self.tree_, self.node_sizes, self.positions, self.max_depth, min_node_spacing=optimal_spacing)
        else:
            TreePositionCalculator.compute_positions(self.tree_, self.node_sizes, self.positions, self.max_depth)

        if ax is None:
            _, ax = plt.subplots(figsize=(12, 8))

        self.plot_tree(ax)
        return ax
