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
Edge label formatter for decision tree visualization.

Provides stateless methods for formatting edge labels in decision trees.
Centralizes all edge/path label logic for better separation of concerns.
"""

from typing import Tuple
from .....utils.feature_metadata_utils import FeatureMetadataUtils
from .....utils.text_utils import TextUtils
from .feature_label_builder import FeatureLabelBuilder


class EdgeLabelFormatter:
    """
    Stateless helper for formatting edge and path labels.

    Centralizes logic for:
    - Edge labels (visual connections between nodes)
    - Path labels (decision path at top of nodes)
    - Symbol formatting (✓/✗ for show_edge_symbols mode)
    - Threshold labels (continuous features)
    - Discrete class labels

    Examples
    --------
    >>> # Get edge labels for visualization
    >>> left, right = EdgeLabelFormatter.get_edge_labels(
    ...     5, 0.5, metadata, False, False, 40
    ... )
    >>> print(left, right)  # ('≤ 0.50', '> 0.50')

    >>> # Get symbols only
    >>> left, right = EdgeLabelFormatter.get_edge_symbols(False)
    >>> print(left, right)  # ('✓', '✗')
    """

    @staticmethod
    def get_edge_symbols(reverse_edges: bool) -> Tuple[str, str]:
        """
        Get edge symbols based on reverse semantics.

        For features where "positive" is undesirable (contacts),
        symbols are reversed.

        Parameters
        ----------
        reverse_edges : bool
            Whether to reverse symbol meanings

        Returns
        -------
        Tuple[str, str]
            (left_symbol, right_symbol)

        Examples
        --------
        >>> # Normal features (distances): small is good
        >>> left, right = EdgeLabelFormatter.get_edge_symbols(False)
        >>> print(left, right)  # ('✓', '✗')

        >>> # Reverse features (contacts): contact is bad
        >>> left, right = EdgeLabelFormatter.get_edge_symbols(True)
        >>> print(left, right)  # ('✗', '✓')

        Notes
        -----
        - False (normal): left=✓ (good), right=✗ (bad)
        - True (reverse): left=✗ (bad), right=✓ (good)
        """
        return ("✗", "✓") if reverse_edges else ("✓", "✗")

    @staticmethod
    def format_path_label_with_symbol(rule: str, reverse_edges: bool, is_left: bool) -> str:
        """
        Format path label with symbol.

        Combines rule text with appropriate symbol for path display.

        Parameters
        ----------
        rule : str
            Split criterion rule (e.g., "ALA25-LEU29" or "distances: ALA25-LEU29 ≤ 4.5 Å")
        reverse_edges : bool
            Whether to reverse symbol meanings
        is_left : bool
            True for left branch, False for right branch

        Returns
        -------
        str
            Formatted path label with symbol

        Examples
        --------
        >>> # Normal feature, left branch
        >>> label = EdgeLabelFormatter.format_path_label_with_symbol(
        ...     "distances: ALA5-LEU10 ≤ 4.50 Å", False, True
        ... )
        >>> print(label)  # "distances: ALA5-LEU10 ≤ 4.50 Å: ✓"

        >>> # Reverse feature, right branch
        >>> label = EdgeLabelFormatter.format_path_label_with_symbol(
        ...     "ALA5-LEU10", True, False
        ... )
        >>> print(label)  # "ALA5-LEU10: ✓"

        Notes
        -----
        Symbols are swapped when reverse_edges=True
        """
        left_symbol, right_symbol = EdgeLabelFormatter.get_edge_symbols(reverse_edges)
        symbol = left_symbol if is_left else right_symbol
        return f"{rule}: {symbol}"

    @staticmethod
    def get_path_labels(
        feature_idx: int,
        threshold: float,
        feature_metadata,
        short_labels: bool,
        hide_feature_type_prefix: bool,
        show_edge_symbols: bool,
        get_discrete_labels_func
    ) -> Tuple[str, str]:
        """
        Get path labels for decision tree visualization.

        Creates labels for the decision path shown at top of child nodes,
        optionally with symbols when show_edge_symbols=True.

        Parameters
        ----------
        feature_idx : int
            Feature index
        threshold : float
            Decision threshold
        feature_metadata : list
            Feature metadata array
        short_labels : bool
            Use short discrete labels
        hide_feature_type_prefix : bool
            Hide feature type prefix
        show_edge_symbols : bool
            Include symbols in path labels
        get_discrete_labels_func : callable
            Function to get discrete labels (from NodeTextFormatter)

        Returns
        -------
        Tuple[str, str]
            (left_path_label, right_path_label)

        Examples
        --------
        >>> # With symbols
        >>> left, right = EdgeLabelFormatter.get_path_labels(
        ...     5, 0.5, metadata, False, False, True, get_discrete_labels
        ... )
        >>> print(left)  # "contacts: ALA5-LEU10: ✓"

        >>> # Without symbols
        >>> left, right = EdgeLabelFormatter.get_path_labels(
        ...     5, 0.5, metadata, False, False, False, get_discrete_labels
        ... )
        >>> print(left)  # "contacts: ALA5-LEU10 = Non-Contact"

        Notes
        -----
        Used by decision tree visualizer for path labels in node boxes
        """
        if show_edge_symbols:
            # Get discrete labels
            discrete_labels = get_discrete_labels_func(
                feature_idx, feature_metadata, short_labels
            )

            # Get split criterion rule
            from .node_text_formatter import NodeTextFormatter
            rule = NodeTextFormatter.get_split_criterion_rule(
                feature_idx, threshold, feature_metadata,
                discrete_labels, hide_feature_type_prefix
            )

            # Get reverse edges setting
            feature_type = FeatureMetadataUtils.get_feature_type(
                feature_metadata, feature_idx
            )
            type_meta = FeatureMetadataUtils.get_top_level_metadata(
                feature_type, feature_metadata
            )
            reverse_edges = FeatureLabelBuilder.get_reverse_edge_semantics(type_meta)

            # Format with symbols
            left_label = EdgeLabelFormatter.format_path_label_with_symbol(
                rule, reverse_edges, is_left=True
            )
            right_label = EdgeLabelFormatter.format_path_label_with_symbol(
                rule, reverse_edges, is_left=False
            )
            return left_label, right_label

        # Use standard edge labels without symbols
        return EdgeLabelFormatter.get_edge_labels(
            feature_idx, threshold, feature_metadata, short_labels,
            short_edge_labels=False, wrap_length=40,
            show_edge_symbols=False, hide_feature_type_prefix=hide_feature_type_prefix,
            get_discrete_labels_func=get_discrete_labels_func
        )

    @staticmethod
    def get_edge_labels(
        feature_idx: int,
        threshold: float,
        feature_metadata,
        short_labels: bool,
        short_edge_labels: bool,
        wrap_length: int,
        show_edge_symbols: bool = False,
        hide_feature_type_prefix: bool = False,
        get_discrete_labels_func=None
    ) -> Tuple[str, str]:
        """
        Get feature-specific edge labels with complete information.

        Creates edge labels containing feature type, feature name, and
        value/threshold information for edges between nodes.

        Parameters
        ----------
        feature_idx : int
            Index of the feature
        threshold : float
            Decision threshold
        feature_metadata : list
            Feature metadata array
        short_labels : bool
            Use short discrete labels
        short_edge_labels : bool
            Use short edge labels
        wrap_length : int
            Maximum line length
        show_edge_symbols : bool, default=False
            Show only symbols (✓/✗)
        hide_feature_type_prefix : bool, default=False
            Hide feature type prefix
        get_discrete_labels_func : callable, optional
            Function to get discrete labels

        Returns
        -------
        Tuple[str, str]
            (left_label, right_label)

        Examples
        --------
        >>> left, right = EdgeLabelFormatter.get_edge_labels(
        ...     5, 0.5, metadata, False, False, 40
        ... )
        >>> print(left, right)  # ('contact: Leu13-ARG31 = NC', 'contact: Leu13-ARG31 = C')
        """
        # Get feature type and name
        feature_type = FeatureMetadataUtils.get_feature_type(feature_metadata, feature_idx)
        feature_name = FeatureMetadataUtils.get_feature_name(feature_metadata, feature_idx)

        # Get top-level metadata
        type_meta = FeatureMetadataUtils.get_top_level_metadata(feature_type, feature_metadata)

        # Get discrete labels
        if get_discrete_labels_func:
            discrete_labels = get_discrete_labels_func(feature_idx, feature_metadata, short_labels)
        else:
            from .node_text_formatter import NodeTextFormatter
            discrete_labels = NodeTextFormatter.get_discrete_labels(feature_idx, feature_metadata, short_labels)

        # If show_edge_symbols, return only symbols
        if show_edge_symbols:
            reverse_edges = FeatureLabelBuilder.get_reverse_edge_semantics(type_meta)
            return EdgeLabelFormatter.get_edge_symbols(reverse_edges)

        # Choose format based on short_edge_labels
        if short_edge_labels:
            return EdgeLabelFormatter.format_short_edge_labels(
                discrete_labels, threshold, type_meta, wrap_length
            )
        return EdgeLabelFormatter.format_default_edge_labels(
            discrete_labels, feature_type, feature_name, threshold,
            type_meta, wrap_length, hide_feature_type_prefix
        )

    @staticmethod
    def format_short_edge_labels(discrete_labels, threshold, type_meta, wrap_length):
        """
        Format short edge labels (value only).

        Parameters
        ----------
        discrete_labels : tuple or None
            Discrete class labels
        threshold : float
            Decision threshold
        type_meta : dict
            Feature type metadata
        wrap_length : int
            Maximum line length

        Returns
        -------
        tuple
            (left_label, right_label) short format

        Examples
        --------
        >>> left, right = EdgeLabelFormatter.format_short_edge_labels(
        ...     ('NC', 'C'), 0.5, {}, 40
        ... )
        >>> print(left, right)  # ('NC', 'C')
        """
        if discrete_labels:
            left = TextUtils.wrap_text(discrete_labels[0], wrap_length)
            right = TextUtils.wrap_text(discrete_labels[1], wrap_length)
            return left, right
        return EdgeLabelFormatter.format_continuous_edge_labels(
            threshold, type_meta, short=True, wrap_length=wrap_length
        )

    @staticmethod
    def format_default_edge_labels(
        discrete_labels, feature_type, feature_name, threshold,
        type_meta, wrap_length, hide_feature_type_prefix=False
    ):
        """
        Format default edge labels (full information).

        Parameters
        ----------
        discrete_labels : tuple or None
            Discrete class labels
        feature_type : str
            Feature type
        feature_name : str
            Feature name
        threshold : float
            Decision threshold
        type_meta : dict
            Feature type metadata
        wrap_length : int
            Maximum line length
        hide_feature_type_prefix : bool, default=False
            Hide feature type prefix

        Returns
        -------
        tuple
            (left_label, right_label) default format

        Examples
        --------
        >>> left, right = EdgeLabelFormatter.format_default_edge_labels(
        ...     ('NC', 'C'), 'contacts', 'ALA5-LEU10', 0.5, {}, 40, False
        ... )
        >>> print(left)  # 'contacts: ALA5-LEU10 = NC'
        """
        if discrete_labels:
            # Check prefix hiding
            allow_hide = type_meta.get('visualization', {}).get('allow_hide_prefix', False) if type_meta else False
            should_hide = hide_feature_type_prefix and allow_hide

            # Build labels
            if should_hide:
                left_line = f"{feature_name} = {discrete_labels[0]}"
                right_line = f"{feature_name} = {discrete_labels[1]}"
            else:
                left_line = f"{feature_type}: {feature_name} = {discrete_labels[0]}"
                right_line = f"{feature_type}: {feature_name} = {discrete_labels[1]}"

            wrapped_left = TextUtils.wrap_text(left_line, wrap_length)
            wrapped_right = TextUtils.wrap_text(right_line, wrap_length)
            return wrapped_left, wrapped_right

        return EdgeLabelFormatter.format_continuous_edge_labels(
            threshold, type_meta, short=False, feature_type=feature_type,
            feature_name=feature_name, wrap_length=wrap_length,
            hide_feature_type_prefix=hide_feature_type_prefix
        )

    @staticmethod
    def format_continuous_edge_labels(
        threshold, type_meta, short=True, feature_type=None,
        feature_name=None, wrap_length=40, hide_feature_type_prefix=False
    ):
        """
        Format continuous edge labels with threshold.

        Parameters
        ----------
        threshold : float
            Decision threshold
        type_meta : dict
            Feature type metadata
        short : bool, default=True
            Use short format
        feature_type : str, optional
            Feature type (required if short=False)
        feature_name : str, optional
            Feature name (required if short=False)
        wrap_length : int, default=40
            Maximum line length
        hide_feature_type_prefix : bool, default=False
            Hide feature type prefix

        Returns
        -------
        tuple
            (left_label, right_label) for continuous features

        Examples
        --------
        >>> left, right = EdgeLabelFormatter.format_continuous_edge_labels(
        ...     4.5, {'units': 'Å'}, short=True
        ... )
        >>> print(left, right)  # ('≤ 4.50 Å', '> 4.50 Å')
        """
        unit_str = EdgeLabelFormatter._get_unit_string(type_meta)

        if short:
            left_line, right_line = EdgeLabelFormatter._format_short_threshold_labels(threshold, unit_str)
        else:
            left_line, right_line = EdgeLabelFormatter._format_full_threshold_labels(
                threshold, unit_str, feature_type, feature_name,
                hide_feature_type_prefix, type_meta
            )

        left = TextUtils.wrap_text(left_line, wrap_length)
        right = TextUtils.wrap_text(right_line, wrap_length)
        return left, right

    @staticmethod
    def _get_unit_string(type_meta):
        """
        Get unit string from metadata.

        Parameters
        ----------
        type_meta : dict
            Feature type metadata

        Returns
        -------
        str
            Unit string with leading space or empty

        Examples
        --------
        >>> unit_str = EdgeLabelFormatter._get_unit_string({'units': 'Å'})
        >>> print(repr(unit_str))  # ' Å'
        """
        unit = type_meta.get("units", "")
        return f" {unit}" if unit else ""

    @staticmethod
    def _format_short_threshold_labels(threshold, unit_str):
        """
        Format short threshold labels.

        Parameters
        ----------
        threshold : float
            Decision threshold
        unit_str : str
            Unit string

        Returns
        -------
        tuple
            (left_label, right_label)

        Examples
        --------
        >>> left, right = EdgeLabelFormatter._format_short_threshold_labels(4.5, ' Å')
        >>> print(left, right)  # '≤ 4.50 Å' '> 4.50 Å'
        """
        left_line = f"≤ {threshold:.2f}{unit_str}"
        right_line = f"> {threshold:.2f}{unit_str}"
        return left_line, right_line

    @staticmethod
    def _format_full_threshold_labels(
        threshold, unit_str, feature_type, feature_name,
        hide_feature_type_prefix, type_meta
    ):
        """
        Format full threshold labels with feature name.

        Parameters
        ----------
        threshold : float
            Decision threshold
        unit_str : str
            Unit string
        feature_type : str
            Feature type
        feature_name : str
            Feature name
        hide_feature_type_prefix : bool
            Hide prefix flag
        type_meta : dict
            Feature type metadata

        Returns
        -------
        tuple
            (left_label, right_label)

        Examples
        --------
        >>> left, right = EdgeLabelFormatter._format_full_threshold_labels(
        ...     4.5, ' Å', 'distances', 'ALA5-LEU10', False, {}
        ... )
        >>> print(left)  # 'distances: ALA5-LEU10 ≤ 4.50 Å'
        """
        allow_hide = type_meta.get('visualization', {}).get('allow_hide_prefix', False) if type_meta else False
        should_hide = hide_feature_type_prefix and allow_hide

        base_label = feature_name if should_hide else f"{feature_type}: {feature_name}"

        left_line = f"{base_label} ≤ {threshold:.2f}{unit_str}"
        right_line = f"{base_label} > {threshold:.2f}{unit_str}"
        return left_line, right_line
