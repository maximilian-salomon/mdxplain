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
Node text formatter for decision tree visualization.

Provides stateless formatting methods for decision tree node text.
All methods are static and receive required state as parameters.
Code is copied 1:1 from DecisionTreeVisualizer for consistency.
"""

from typing import Optional, Tuple

from .....utils.feature_metadata_utils import FeatureMetadataUtils
from .....utils.text_utils import TextUtils
from .decision_tree_visualization_config import DecisionTreeVisualizationConfig
from .feature_label_builder import FeatureLabelBuilder
TREE_CONFIG = DecisionTreeVisualizationConfig()


class NodeTextFormatter:
    """
    Stateless formatter for decision tree node text.

    All methods are static and receive required state as parameters.
    Code is copied 1:1 from DecisionTreeVisualizer for consistency.

    Examples
    --------
    >>> text = NodeTextFormatter.format_root_node_text(
    ...     [100, 50], class_names, total_counts, 'distances', 'Leu13-ARG31',
    ...     3.5, None
    ... )
    """

    @staticmethod
    def shorten_class_name(name: str, max_length: int = 16) -> str:
        """
        Shorten class name to max_length with [...] pattern.

        Parameters
        ----------
        name : str
            Class name to shorten
        max_length : int, default=16
            Maximum length for class name

        Returns
        -------
        str
            Shortened class name with format first_7[...]last_7

        Examples
        --------
        >>> NodeTextFormatter.shorten_class_name("very_long_class_name", 16)
        'very_lo[...]name'

        >>> NodeTextFormatter.shorten_class_name("short", 16)
        'short'
        """
        if len(name) <= max_length:
            return name
        half = (max_length // 2) - 5
        return f"{name[:half]}[...]{name[-half:]}"

    @staticmethod
    def get_discrete_labels(feature_idx: int, feature_metadata, short_labels: bool) -> Optional[Tuple[str, str]]:
        """
        Get discrete labels for a feature if applicable.

        Checks if the feature is discrete (using visualization.is_discrete flag)
        and returns the class labels for edge and node labeling.

        Parameters
        ----------
        feature_idx : int
            Index of the feature in the tree
        feature_metadata : list
            Feature metadata array
        short_labels : bool
            Whether to use short labels

        Returns
        -------
        Tuple[str, str] or None
            (left_label, right_label) if discrete feature, None if continuous

        Examples
        --------
        >>> labels = NodeTextFormatter.get_discrete_labels(5, metadata, False)
        >>> print(labels)  # ("Non-Contact", "Contact")

        >>> labels = NodeTextFormatter.get_discrete_labels(10, metadata, False)  # DSSP feature
        >>> print(labels)  # ("H", "E") or similar

        Notes
        -----
        Uses type_metadata["visualization"]["is_discrete"] to detect discrete
        features (DSSP, contacts, etc.), NOT the threshold value!
        """
        # Get feature_type for this feature
        feature_type = FeatureMetadataUtils.get_feature_type(
            feature_metadata, feature_idx
        )

        # Get top-level metadata for this TYPE
        type_meta = FeatureMetadataUtils.get_top_level_metadata(
            feature_type, feature_metadata
        )

        # Check if feature is discrete using the visualization flag
        viz = type_meta.get("visualization", {})
        if not viz.get("is_discrete", False):
            return None  # Not a discrete feature

        # Extract classes from visualization.tick_labels
        tick_labels_dict = viz.get("tick_labels", {})

        # Priority based on short_labels setting
        if short_labels:
            # Try short first, then long as fallback
            classes = tick_labels_dict.get("short", tick_labels_dict.get("long"))
        else:
            # Default: Try long first, then short as fallback
            classes = tick_labels_dict.get("long", tick_labels_dict.get("short"))

        if classes and len(classes) >= 2:
            # Return original strings (wrapping happens in _format_* methods)
            return (classes[0], classes[1])

        return None

    @staticmethod
    def format_node_text(path, counts, percentages, feature_type,
                          feature_name, threshold, discrete_labels,
                          class_names, total_counts, hide_path, wrap_length,
                          hide_node_frames=False, hide_feature_type_prefix=False,
                          feature_metadata=None, feature_idx=None):
        """
        Format text for decision node (root or internal).

        Parameters
        ----------
        path : list of str
            Decision path to this node
        counts : list of int
            Class counts
        percentages : list of float
            Class percentages
        feature_type : str
            Feature type
        feature_name : str
            Feature name
        threshold : float
            Decision threshold
        discrete_labels : tuple or None
            Discrete labels if applicable
        class_names : list of str
            Class names
        total_counts : list of int
            Total counts per class
        short_layout : bool
            Whether using short layout
        wrap_length : int
            Maximum line length for text wrapping
        hide_node_frames : bool, default=False
            Hide frame counts in non-root nodes
        hide_feature_type_prefix : bool, default=False
            Hide feature type prefix in labels
        feature_metadata : list, optional
            Feature metadata for accessing type-level metadata
        feature_idx : int, optional
            Feature index for accessing type-level metadata

        Returns
        -------
        str
            Formatted node text
        """
        if not path:
            return NodeTextFormatter.format_root_node_text(
                counts, class_names, total_counts, feature_type, feature_name, threshold,
                discrete_labels, wrap_length, hide_feature_type_prefix, feature_metadata, feature_idx
            )
        return NodeTextFormatter.format_decision_node_text(
            path, counts, percentages, class_names, total_counts, feature_type, feature_name,
            threshold, discrete_labels, hide_path, wrap_length, hide_node_frames,
            hide_feature_type_prefix, feature_metadata, feature_idx
        )

    @staticmethod
    def format_root_node_text(counts, class_names, total_counts, feature_type, feature_name, threshold, discrete_labels, wrap_length, hide_feature_type_prefix=False, feature_metadata=None, feature_idx=None):
        """
        Format text for root node.

        Creates multi-line text displaying class counts and split feature.

        Parameters
        ----------
        counts : list of int
            Class sample counts for this node
        class_names : list of str
            Class names
        total_counts : list of int
            Total counts per class
        feature_type : str
            Type of feature (e.g., 'distances', 'contacts')
        feature_name : str
            Name of feature (e.g., 'Leu13-ARG31')
        threshold : float
            Decision threshold value
        discrete_labels : tuple of str or None
            Discrete class labels if applicable
        wrap_length : int
            Maximum line length for text wrapping
        hide_feature_type_prefix : bool, default=False
            Hide feature type prefix in labels
        feature_metadata : list, optional
            Feature metadata for accessing type-level metadata
        feature_idx : int, optional
            Feature index for accessing type-level metadata

        Returns
        -------
        str
            Formatted text for root node

        Examples
        --------
        >>> text = NodeTextFormatter.format_root_node_text(
        ...     [100, 50], ['Class1', 'Class2'], [100, 50], 'distances', 'Leu13-ARG31', 3.5, None, 40
        ... )
        >>> print(text)
        Class1: 100 / 100
        Class2: 50 / 50

        distances: Leu13-ARG31 <= 3.50
        """
        text = ""
        for i, class_name in enumerate(class_names):
            line = f"{class_name}: {counts[i]} / {total_counts[i]}"
            wrapped_line = TextUtils.wrap_text(line, wrap_length)
            text += f"{wrapped_line}\n"

        # Build complete feature label (prefix + suffix)
        feature_line = FeatureLabelBuilder.build_feature_label(
            feature_type, feature_name, threshold, discrete_labels,
            hide_feature_type_prefix, feature_metadata, feature_idx
        )

        wrapped_feature_line = TextUtils.wrap_text(feature_line, wrap_length)
        text += f"\n{wrapped_feature_line}"

        return text

    @staticmethod
    def format_path_section(path, hide_path, wrap_length):
        """
        Format decision path section for node text.

        Parameters
        ----------
        path : list of str
            Decision path from root to this node
        hide_path : bool
            Whether to hide the path display
        wrap_length : int
            Maximum line length for text wrapping

        Returns
        -------
        str
            Formatted path text or empty string if hide_path is True

        Examples
        --------
        >>> text = NodeTextFormatter.format_path_section(['Contact', 'State A'], False, 40)
        >>> print(text)
        Contact
        State A
        """
        if not hide_path:
            wrapped_path = [TextUtils.wrap_text(p, wrap_length) for p in path]
            path_text = '\n'.join(wrapped_path)
            return f"{path_text}\n"
        return ""

    @staticmethod
    def format_class_statistics_section(counts, percentages, class_names, total_counts, wrap_length, hide_node_frames=False):
        """
        Format class statistics section for node text.

        Parameters
        ----------
        counts : list of int
            Class sample counts for this node
        percentages : list of float
            Percentage of total samples per class
        class_names : list of str
            Class names
        total_counts : list of int
            Total counts per class
        wrap_length : int
            Maximum line length for text wrapping
        hide_node_frames : bool, default=False
            Hide frame counts, showing only percentages

        Returns
        -------
        str
            Formatted class statistics text

        Examples
        --------
        >>> text = NodeTextFormatter.format_class_statistics_section(
        ...     [80, 20], [80.0, 40.0], ['Class1', 'Class2'], [100, 50], 40
        ... )
        >>> print(text)
        Class1: 80 / 100 (80.0%)
        Class2: 20 / 50 (40.0%)

        >>> text = NodeTextFormatter.format_class_statistics_section(
        ...     [80, 20], [80.0, 40.0], ['Class1', 'Class2'], [100, 50], 40, hide_node_frames=True
        ... )
        >>> print(text)
        Class1: 80.0%
        Class2: 40.0%
        """
        text = ""
        for i, class_name in enumerate(class_names):
            if hide_node_frames:
                line = f"{class_name}: {percentages[i]:.1f}%"
            else:
                line = f"{class_name}: {counts[i]} / {total_counts[i]} ({percentages[i]:.1f}%)"
            wrapped_line = TextUtils.wrap_text(line, wrap_length)
            text += f"{wrapped_line}\n"
        return text

    @staticmethod
    def format_split_criterion_section(feature_type, feature_name,
                                        threshold, discrete_labels, wrap_length,
                                        hide_feature_type_prefix=False, feature_metadata=None, feature_idx=None):
        """
        Format splitting criterion section for node text.

        Parameters
        ----------
        feature_type : str
            Type of feature (e.g., 'distances', 'contacts')
        feature_name : str
            Name of feature (e.g., 'Leu13-ARG31')
        threshold : float
            Decision threshold value
        discrete_labels : tuple of str or None
            Discrete class labels if applicable
        wrap_length : int
            Maximum line length for text wrapping
        hide_feature_type_prefix : bool, default=False
            Hide feature type prefix in labels
        feature_metadata : list, optional
            Feature metadata for accessing type-level metadata
        feature_idx : int, optional
            Feature index for accessing type-level metadata

        Returns
        -------
        str
            Formatted splitting criterion text

        Examples
        --------
        >>> text = NodeTextFormatter.format_split_criterion_section(
        ...     'distances', 'Leu13-ARG31', 3.5, None, 40
        ... )
        >>> print(text)
        distances: Leu13-ARG31 <= 3.50
        """
        # Build complete feature label (prefix + suffix)
        criterion_line = FeatureLabelBuilder.build_feature_label(
            feature_type, feature_name, threshold, discrete_labels,
            hide_feature_type_prefix, feature_metadata, feature_idx
        )

        wrapped_criterion = TextUtils.wrap_text(criterion_line, wrap_length)
        return f"\n{wrapped_criterion}"

    @staticmethod
    def format_decision_node_text(path, counts, percentages, class_names, total_counts, feature_type,
                                    feature_name, threshold, discrete_labels, hide_path, wrap_length,
                                    hide_node_frames=False, hide_feature_type_prefix=False,
                                    feature_metadata=None, feature_idx=None):
        """
        Format text for decision node.

        Creates multi-line text with decision path, class statistics, and split feature.

        Parameters
        ----------
        path : list of str
            Decision path from root to this node
        counts : list of int
            Class sample counts for this node
        percentages : list of float
            Percentage of total samples per class
        class_names : list of str
            Class names
        total_counts : list of int
            Total counts per class
        feature_type : str
            Type of feature (e.g., 'distances', 'contacts')
        feature_name : str
            Name of feature (e.g., 'Leu13-ARG31')
        threshold : float
            Decision threshold value
        discrete_labels : tuple of str or None
            Discrete class labels if applicable
        short_layout : bool
            Whether using short layout
        wrap_length : int
            Maximum line length for text wrapping
        hide_node_frames : bool, default=False
            Hide frame counts in non-root nodes
        hide_feature_type_prefix : bool, default=False
            Hide feature type prefix in labels
        feature_metadata : list, optional
            Feature metadata for accessing type-level metadata
        feature_idx : int, optional
            Feature index for accessing type-level metadata

        Returns
        -------
        str
            Formatted text for decision node

        Examples
        --------
        >>> text = NodeTextFormatter.format_decision_node_text(
        ...     ['Contact'], [80, 20], [80.0, 40.0], ['C1', 'C2'], [100, 50],
        ...     'distances', 'Leu13-ARG31', 3.5, None, False, 40
        ... )
        >>> print(text)
        Contact
        Class1: 80 / 100 (80.0%)
        Class2: 20 / 50 (40.0%)

        distances: Leu13-ARG31 <= 3.50
        """
        path_text = NodeTextFormatter.format_path_section(path, hide_path, wrap_length)
        stats_text = NodeTextFormatter.format_class_statistics_section(
            counts, percentages, class_names, total_counts, wrap_length, hide_node_frames
        )
        criterion_text = NodeTextFormatter.format_split_criterion_section(
            feature_type, feature_name, threshold, discrete_labels, wrap_length,
            hide_feature_type_prefix, feature_metadata, feature_idx
        )
        return path_text + stats_text + criterion_text

    @staticmethod
    def format_leaf_node_text(path, counts, percentages, class_names, total_counts, short_layout, wrap_length, hide_node_frames=False):
        """
        Format text for leaf node.

        Creates multi-line text with decision path and final class statistics.

        Parameters
        ----------
        path : list of str
            Decision path from root to this leaf
        counts : list of int
            Class sample counts for this leaf
        percentages : list of float
            Percentage of total samples per class
        class_names : list of str
            Class names
        total_counts : list of int
            Total counts per class
        short_layout : bool
            Whether using short layout
        wrap_length : int
            Maximum line length for text wrapping
        hide_node_frames : bool, default=False
            Hide frame counts, showing only percentages

        Returns
        -------
        str
            Formatted text for leaf node

        Examples
        --------
        >>> text = NodeTextFormatter.format_leaf_node_text(
        ...     ['Contact', 'distances: Leu13-ARG31 <= 3.5'], [60, 5], [60.0, 10.0],
        ...     ['Class1', 'Class2'], [100, 50], False, 40
        ... )
        >>> print(text)
        Contact
        distances: Leu13-ARG31 <= 3.5
        Class1: 60 / 100 (60.0%)
        Class2: 5 / 50 (10.0%)
        """
        # Skip path if short_layout is enabled
        if not short_layout:
            wrapped_path = [TextUtils.wrap_text(p, wrap_length) for p in path]
            path_text = '\n'.join(wrapped_path)
            text = f"{path_text}\n"
        else:
            text = ""

        for i, class_name in enumerate(class_names):
            if hide_node_frames:
                line = f"{class_name}: {percentages[i]:.1f}%"
            else:
                line = f"{class_name}: {counts[i]} / {total_counts[i]} ({percentages[i]:.1f}%)"
            wrapped_line = TextUtils.wrap_text(line, wrap_length)
            text += f"{wrapped_line}\n"
        return text

    @staticmethod
    def should_hide_prefix(feature_type: str, feature_metadata, hide_feature_type_prefix: bool) -> bool:
        """
        Check if feature type prefix should be hidden based on metadata and parameter.

        Prefix is only hidden if both conditions are met:
        1. hide_feature_type_prefix parameter is True
        2. Feature type allows hiding via 'allow_hide_prefix' in metadata

        Parameters
        ----------
        feature_type : str
            Feature type (e.g., 'contacts', 'distances')
        feature_metadata : list
            Feature metadata array
        hide_feature_type_prefix : bool
            User parameter requesting prefix hiding

        Returns
        -------
        bool
            True if prefix should be hidden, False otherwise

        Examples
        --------
        >>> # Contacts with allow_hide_prefix=True
        >>> should_hide = NodeTextFormatter.should_hide_prefix('contacts', metadata, True)
        >>> print(should_hide)
        True

        >>> # Distances without allow_hide_prefix
        >>> should_hide = NodeTextFormatter.should_hide_prefix('distances', metadata, True)
        >>> print(should_hide)
        False
        """
        if not hide_feature_type_prefix:
            return False

        # Get type-level metadata
        type_meta = FeatureMetadataUtils.get_top_level_metadata(feature_type, feature_metadata)

        # Check if prefix hiding is allowed via metadata
        if type_meta:
            vis_metadata = type_meta.get('visualization', {})
            return vis_metadata.get('allow_hide_prefix', False)

        return False

    @staticmethod
    def get_split_criterion_rule(feature_idx: int, threshold: float, feature_metadata,
                                   discrete_labels, hide_feature_type_prefix: bool = False) -> str:
        """
        Get the split criterion rule for a decision tree node.

        Extracts only the rule portion without class statistics, used for path labels
        when show_edge_symbols is enabled. Returns the feature description with
        threshold information for continuous features or just the feature name for
        discrete features.

        Parameters
        ----------
        feature_idx : int
            Index of the feature in the tree
        threshold : float
            Decision threshold from tree
        feature_metadata : list
            Feature metadata array
        discrete_labels : tuple or None
            Discrete class labels if applicable (not used in output)
        hide_feature_type_prefix : bool, default=False
            Hide feature type prefix in labels (only if allow_hide_prefix is True)

        Returns
        -------
        str
            Split criterion rule (e.g., "ALA25-LEU29" or "distances: ALA25-LEU29 ≤ 4.5 Å")

        Examples
        --------
        >>> # Discrete feature (contacts)
        >>> rule = NodeTextFormatter.get_split_criterion_rule(5, 0.5, metadata, ('NC', 'C'), False)
        >>> print(rule)
        'contacts: ALA25-LEU29'

        >>> # Discrete feature with hide_prefix
        >>> rule = NodeTextFormatter.get_split_criterion_rule(5, 0.5, metadata, ('NC', 'C'), True)
        >>> print(rule)
        'ALA25-LEU29'

        >>> # Continuous feature (distances)
        >>> rule = NodeTextFormatter.get_split_criterion_rule(10, 4.5, metadata, None, False)
        >>> print(rule)
        'distances: ALA25-LEU29 ≤ 4.50 Å'
        """
        # Get feature type and name
        feature_type = FeatureMetadataUtils.get_feature_type(feature_metadata, feature_idx)
        feature_name = FeatureMetadataUtils.get_feature_name(feature_metadata, feature_idx)

        # Build complete feature label with units
        return FeatureLabelBuilder.build_feature_label(
            feature_type, feature_name, threshold, discrete_labels,
            hide_feature_type_prefix, feature_metadata, feature_idx,
            include_units=True
        )
