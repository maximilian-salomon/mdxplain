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
Tree position calculator for decision tree visualization.

Provides stateless position calculation methods for decision tree layout.
All methods are static and receive required state as parameters.
Code is copied 1:1 from DecisionTreeVisualizer for consistency.
"""

import matplotlib.pyplot as plt
from sklearn.tree import _tree

from .decision_tree_visualization_config import DecisionTreeVisualizationConfig

TREE_CONFIG = DecisionTreeVisualizationConfig()


class TreePositionCalculator:
    """
    Stateless calculator for decision tree node positions.

    All methods are static and receive required state as parameters.
    Code is copied 1:1 from DecisionTreeVisualizer for consistency.

    Examples
    --------
    >>> positions = {}
    >>> TreePositionCalculator.compute_positions(
    ...     tree_, node_sizes, positions, max_depth, min_node_spacing=100
    ... )
    """

    @staticmethod
    def calculate_optimal_spacing(tree_, node_sizes, target_width_pixels):
        """
        Calculate optimal node spacing to fit tree in target width.

        Called after build_graph() when node_sizes are known.
        Calculates min_node_spacing that fits all leaf nodes horizontally
        within the target width (only leafs matter for horizontal width).

        Parameters
        ----------
        tree\_ : sklearn.tree._tree.Tree
            The tree structure
        node_sizes : dict
            Dictionary mapping node indices to (width, height) tuples
        target_width_pixels : float
            Target width in pixels (or tree coordinates)

        Returns
        -------
        float
            Optimal min_node_spacing value

        Examples
        --------
        >>> optimal_spacing = TreePositionCalculator.calculate_optimal_spacing(
        ...     tree_, node_sizes, 10000
        ... )
        >>> print(f"Optimal spacing: {optimal_spacing:.1f}px")
        Optimal spacing: 85.3px
        """
        # Identify leaf nodes only (these are all on the same horizontal level)
        leaf_nodes = []
        for node in node_sizes.keys():
            if tree_.feature[node] == _tree.TREE_UNDEFINED:
                leaf_nodes.append(node)

        # Sum only leaf node widths
        total_leaf_width = sum(node_sizes[node][0] for node in leaf_nodes)

        # Count gaps between leafs
        n_leafs = len(leaf_nodes)
        n_gaps = n_leafs - 1

        if n_gaps == 0:
            return TREE_CONFIG.optimal_spacing_single_node

        # Available space for gaps
        available_space = target_width_pixels - total_leaf_width

        # Calculate spacing per gap
        optimal_spacing = available_space / n_gaps

        # Clamp to reasonable range (allow tighter for deep trees)
        return max(
            TREE_CONFIG.optimal_spacing_min,
            min(TREE_CONFIG.optimal_spacing_max, optimal_spacing)
        )

    @staticmethod
    def get_text_size(text, fontsize, node_pad):
        r"""
        Compute text box size.

        Creates temporary text object to measure bounding box dimensions
        for node sizing calculations.

        Parameters
        ----------
        text : str
            Text content to measure
        fontsize : float
            Font size for text
        node_pad : float
            Padding around node text

        Returns
        -------
        tuple of float
            (width, height) in pixels

        Examples
        --------
        >>> width, height = TreePositionCalculator.get_text_size(
        ...     "Class1: 100\nClass2: 50", 10, 0.5
        ... )
        >>> print(f"Width: {width:.1f}px, Height: {height:.1f}px")
        Width: 150.5px, Height: 45.2px
        """
        fig = plt.figure()
        renderer = fig.canvas.get_renderer()
        t = plt.text(0, 0, text, fontsize=fontsize, ha='center', va='center',
                     bbox=dict(boxstyle=f'round,pad={node_pad}'))
        bb = t.get_window_extent(renderer=renderer)
        width = bb.width
        height = bb.height
        plt.close(fig)
        return width, height

    @staticmethod
    def compute_positions(tree_, node_sizes, positions, max_depth, min_node_spacing=None):
        """
        Compute node positions using in-order traversal with cumulative spacing.

        Creates pyramid structure by assigning x-positions based on actual
        node widths. Each node gets positioned based on previous nodes' widths
        plus minimum spacing, preventing any overlap.

        Parameters
        ----------
        tree\_ : sklearn.tree._tree.Tree
            The tree structure
        node_sizes : dict
            Dictionary mapping node indices to (width, height) tuples
        positions : dict
            Dictionary to store positions (will be modified)
        max_depth : int
            Maximum depth of tree
        min_node_spacing : float, optional
            Minimum spacing between nodes. If None, calculated based on depth.
            Use this for adaptive width control.

        Returns
        -------
        None
            Populates positions dict with x-coordinates
        """
        state = {'current_x': 0}

        # Node spacing: use provided value or calculate based on depth
        if min_node_spacing is None:
            # Node spacing: tighter for deep trees to fit more leafs horizontally
            if max_depth == 6:
                min_node_spacing = TREE_CONFIG.default_spacing_depth_6
            elif max_depth == 5:
                min_node_spacing = TREE_CONFIG.default_spacing_depth_5
            else:
                min_node_spacing = TREE_CONFIG.default_spacing_fallback

        TreePositionCalculator._in_order_traverse(0, min_node_spacing, tree_, node_sizes, positions, state)

    @staticmethod
    def _in_order_traverse(node, min_node_spacing, tree_, node_sizes, positions, state):
        """
        Perform in-order traversal to assign x-positions to nodes.

        Recursively traverses left subtree, assigns position to current node,
        then traverses right subtree. This creates balanced horizontal spacing.

        Parameters
        ----------
        node : int
            Current node index in tree
        min_node_spacing : float
            Minimum spacing between adjacent nodes
        tree_ : sklearn.tree._tree.Tree
            The tree structure
        node_sizes : dict
            Dictionary mapping node indices to (width, height) tuples
        positions : dict
            Dictionary to store positions (will be modified)
        state : dict
            State dictionary containing 'current_x'

        Returns
        -------
        None
            Updates positions dict and state['current_x']
        """
        if node not in node_sizes:
            return

        TreePositionCalculator._traverse_left_subtree(node, min_node_spacing, tree_, node_sizes, positions, state)
        TreePositionCalculator._assign_node_position(node, min_node_spacing, node_sizes, positions, state)
        TreePositionCalculator._traverse_right_subtree(node, min_node_spacing, tree_, node_sizes, positions, state)

    @staticmethod
    def _traverse_left_subtree(node, min_node_spacing, tree_, node_sizes, positions, state):
        """
        Traverse left subtree if node has children.

        Parameters
        ----------
        node : int
            Current node index
        min_node_spacing : float
            Minimum spacing between nodes
        tree_ : sklearn.tree._tree.Tree
            The tree structure
        node_sizes : dict
            Dictionary mapping node indices to (width, height) tuples
        positions : dict
            Dictionary to store positions
        state : dict
            State dictionary containing 'current_x'

        Returns
        -------
        None
            Recursively calls _in_order_traverse on left child
        """
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            left_child = tree_.children_left[node]
            if left_child in node_sizes:
                TreePositionCalculator._in_order_traverse(left_child, min_node_spacing, tree_, node_sizes, positions, state)

    @staticmethod
    def _assign_node_position(node, min_node_spacing, node_sizes, positions, state):
        """
        Assign x-position to node and advance current_x.

        Parameters
        ----------
        node : int
            Current node index
        min_node_spacing : float
            Minimum spacing between nodes
        node_sizes : dict
            Dictionary mapping node indices to (width, height) tuples
        positions : dict
            Dictionary to store positions (will be modified)
        state : dict
            State dictionary containing 'current_x' (will be modified)

        Returns
        -------
        None
            Updates positions and state['current_x']
        """
        positions[node] = {'x': state['current_x']}
        node_width = node_sizes[node][0]
        state['current_x'] += node_width + min_node_spacing

    @staticmethod
    def _traverse_right_subtree(node, min_node_spacing, tree_, node_sizes, positions, state):
        """
        Traverse right subtree if node has children.

        Parameters
        ----------
        node : int
            Current node index
        min_node_spacing : float
            Minimum spacing between nodes
        tree_ : sklearn.tree._tree.Tree
            The tree structure
        node_sizes : dict
            Dictionary mapping node indices to (width, height) tuples
        positions : dict
            Dictionary to store positions
        state : dict
            State dictionary containing 'current_x'

        Returns
        -------
        None
            Recursively calls _in_order_traverse on right child
        """
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            right_child = tree_.children_right[node]
            if right_child in node_sizes:
                TreePositionCalculator._in_order_traverse(right_child, min_node_spacing, tree_, node_sizes, positions, state)
