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
Feature label builder for decision tree visualization.

Provides stateless methods for building feature labels with prefix handling
and rule suffix logic. Eliminates code duplication across formatting methods.
"""

from typing import Optional, Tuple
from .....utils.feature_metadata_utils import FeatureMetadataUtils


class FeatureLabelBuilder:
    """
    Stateless helper for building feature labels.

    Centralizes logic for:
    - Feature prefix handling (show/hide based on metadata)
    - Rule suffix application (discrete labels, thresholds)
    - Metadata extraction for visualization settings

    Examples
    --------
    >>> # Build complete label with prefix and suffix
    >>> label = FeatureLabelBuilder.build_feature_label(
    ...     'distances', 'ALA5-LEU10', 4.5, None, False, metadata, 0
    ... )
    >>> print(label)  # "distances: ALA5-LEU10 <= 4.50"

    >>> # Build label with prefix hidden
    >>> label = FeatureLabelBuilder.build_feature_label(
    ...     'torsions', 'ALA5_phi', 0.5, None, True, metadata, 0
    ... )
    >>> print(label)  # "ALA5_phi"
    """

    @staticmethod
    def build_feature_label(
        feature_type: str,
        feature_name: str,
        threshold: float,
        discrete_labels: Optional[Tuple[str, str]],
        hide_feature_type_prefix: bool,
        feature_metadata,
        feature_idx: Optional[int],
        include_units: bool = False
    ) -> str:
        """
        Build complete feature label with prefix and rule suffix.

        Combines base label (with/without prefix) and rule suffix
        (discrete label or threshold) into final feature label string.

        Parameters
        ----------
        feature_type : str
            Feature type (e.g., 'distances', 'contacts')
        feature_name : str
            Feature name (e.g., 'ALA5-LEU10')
        threshold : float
            Decision threshold value
        discrete_labels : tuple of str or None
            Discrete class labels if applicable
        hide_feature_type_prefix : bool
            Hide feature type prefix if allowed by metadata
        feature_metadata : list
            Feature metadata array
        feature_idx : int or None
            Feature index for metadata lookup
        include_units : bool, default=False
            Include unit string for continuous features

        Returns
        -------
        str
            Complete feature label with prefix and suffix

        Examples
        --------
        >>> # Continuous feature with threshold
        >>> label = FeatureLabelBuilder.build_feature_label(
        ...     'distances', 'ALA5-LEU10', 4.5, None, False, metadata, 0
        ... )
        >>> print(label)  # "distances: ALA5-LEU10 <= 4.50"

        >>> # Discrete feature with custom suffix
        >>> label = FeatureLabelBuilder.build_feature_label(
        ...     'contacts', 'ALA5-LEU10', 0.5, ('NC', 'C'), True, metadata, 0
        ... )
        >>> print(label)  # "ALA5-LEU10"  (no suffix for contacts)

        >>> # Continuous feature with units
        >>> label = FeatureLabelBuilder.build_feature_label(
        ...     'distances', 'ALA5-LEU10', 4.5, None, False, metadata, 0, True
        ... )
        >>> print(label)  # "distances: ALA5-LEU10 ≤ 4.50 Å"

        Notes
        -----
        - Handles both continuous and discrete features
        - Respects metadata-based prefix hiding rules
        - Applies custom rule suffixes from metadata
        - Units only added for continuous features when include_units=True
        """
        base_label = FeatureLabelBuilder.build_base_label(
            feature_type, feature_name, hide_feature_type_prefix, feature_metadata
        )
        return FeatureLabelBuilder.add_rule_suffix(
            base_label, threshold, discrete_labels, feature_type,
            feature_metadata, feature_idx, include_units
        )

    @staticmethod
    def build_base_label(
        feature_type: str,
        feature_name: str,
        hide_feature_type_prefix: bool,
        feature_metadata
    ) -> str:
        """
        Build base feature label with or without prefix.

        Constructs feature label string with feature type prefix
        shown/hidden based on user setting and metadata rules.

        Parameters
        ----------
        feature_type : str
            Feature type (e.g., 'distances', 'contacts')
        feature_name : str
            Feature name (e.g., 'ALA5-LEU10')
        hide_feature_type_prefix : bool
            User setting to hide prefix
        feature_metadata : list
            Feature metadata array

        Returns
        -------
        str
            Base feature label (prefix shown/hidden)

        Examples
        --------
        >>> # With prefix
        >>> label = FeatureLabelBuilder.build_base_label(
        ...     'distances', 'ALA5-LEU10', False, metadata
        ... )
        >>> print(label)  # "distances: ALA5-LEU10"

        >>> # Without prefix (if allowed by metadata)
        >>> label = FeatureLabelBuilder.build_base_label(
        ...     'torsions', 'ALA5_phi', True, metadata
        ... )
        >>> print(label)  # "ALA5_phi"

        Notes
        -----
        - Prefix only hidden if BOTH conditions met:
          1. User requests hiding (hide_feature_type_prefix=True)
          2. Metadata allows hiding (allow_hide_prefix=True)
        """
        should_hide = FeatureLabelBuilder._should_hide_prefix(
            feature_type, feature_metadata, hide_feature_type_prefix
        )

        if should_hide:
            return feature_name
        return f"{feature_type}: {feature_name}"

    @staticmethod
    def get_rule_suffix_from_metadata(
        feature_type: str,
        feature_metadata,
        feature_idx: Optional[int]
    ) -> Optional[str]:
        """
        Get rule suffix from feature type metadata.

        Extracts custom rule suffix setting from visualization metadata.
        Used for features like contacts that have special suffix rules.

        Parameters
        ----------
        feature_type : str
            Feature type to get suffix for
        feature_metadata : list
            Feature metadata array
        feature_idx : int or None
            Feature index (None skips metadata lookup)

        Returns
        -------
        str or None
            Custom rule suffix, or None if not defined

        Examples
        --------
        >>> # Contacts have empty string suffix (no suffix)
        >>> suffix = FeatureLabelBuilder.get_rule_suffix_from_metadata(
        ...     'contacts', metadata, 0
        ... )
        >>> print(repr(suffix))  # ''

        >>> # Distances have no custom suffix
        >>> suffix = FeatureLabelBuilder.get_rule_suffix_from_metadata(
        ...     'distances', metadata, 0
        ... )
        >>> print(suffix)  # None

        Notes
        -----
        - Returns None if feature_idx is None
        - Empty string ('') means no suffix at all
        - None means use default suffix behavior
        """
        if feature_metadata is None or feature_idx is None:
            return None

        type_metadata = FeatureMetadataUtils.get_top_level_metadata(
            feature_type, feature_metadata
        )
        if type_metadata:
            vis_metadata = type_metadata.get('visualization', {})
            return vis_metadata.get('short_rule_suffix', None)
        return None

    @staticmethod
    def add_rule_suffix(
        base_label: str,
        threshold: float,
        discrete_labels: Optional[Tuple[str, str]],
        feature_type: str,
        feature_metadata,
        feature_idx: Optional[int],
        include_units: bool = False
    ) -> str:
        """
        Add rule suffix to base feature label.

        Appends discrete class label or threshold to base label,
        respecting custom rule suffix settings from metadata.

        Parameters
        ----------
        base_label : str
            Base feature label (from build_base_label)
        threshold : float
            Decision threshold value
        discrete_labels : tuple of str or None
            Discrete class labels if applicable
        feature_type : str
            Feature type for metadata lookup
        feature_metadata : list
            Feature metadata array
        feature_idx : int or None
            Feature index for metadata lookup
        include_units : bool, default=False
            Include unit string for continuous features

        Returns
        -------
        str
            Feature label with rule suffix applied

        Examples
        --------
        >>> # Discrete feature with default suffix
        >>> label = FeatureLabelBuilder.add_rule_suffix(
        ...     'distances: ALA5-LEU10', 0.5, ('NC', 'C'), 'distances', metadata, 0
        ... )
        >>> print(label)  # "distances: ALA5-LEU10 = C"

        >>> # Continuous feature with threshold
        >>> label = FeatureLabelBuilder.add_rule_suffix(
        ...     'distances: ALA5-LEU10', 4.5, None, 'distances', metadata, 0
        ... )
        >>> print(label)  # "distances: ALA5-LEU10 <= 4.50"

        >>> # Continuous feature with units
        >>> label = FeatureLabelBuilder.add_rule_suffix(
        ...     'distances: ALA5-LEU10', 4.5, None, 'distances', metadata, 0, True
        ... )
        >>> print(label)  # "distances: ALA5-LEU10 ≤ 4.50 Å"

        Notes
        -----
        - Discrete features: Check for custom short_rule_suffix
        - Empty string suffix means no suffix at all
        - Continuous features: Add threshold if != 0.5
        - Units only added for continuous features when include_units=True
        """
        if discrete_labels:
            return FeatureLabelBuilder._add_discrete_suffix(
                base_label, discrete_labels, feature_type, feature_metadata, feature_idx
            )

        return FeatureLabelBuilder._add_continuous_suffix(
            base_label, threshold, feature_type, feature_metadata, include_units
        )

    @staticmethod
    def _add_discrete_suffix(base_label, discrete_labels, feature_type, feature_metadata, feature_idx):
        """
        Add discrete feature suffix.

        Parameters
        ----------
        base_label : str
            Base label
        discrete_labels : tuple
            Discrete labels
        feature_type : str
            Feature type
        feature_metadata : list
            Metadata array
        feature_idx : int or None
            Feature index

        Returns
        -------
        str
            Label with discrete suffix

        Examples
        --------
        >>> label = FeatureLabelBuilder._add_discrete_suffix(
        ...     'contacts: ALA5-LEU10', ('NC', 'C'), 'contacts', metadata, 0
        ... )
        >>> print(label)  # 'contacts: ALA5-LEU10'
        """
        short_rule_suffix = FeatureLabelBuilder.get_rule_suffix_from_metadata(
            feature_type, feature_metadata, feature_idx
        )

        if short_rule_suffix == '':
            return base_label
        if short_rule_suffix is not None:
            return f"{base_label}{short_rule_suffix}"
        return f"{base_label} = {discrete_labels[1]}"

    @staticmethod
    def _add_continuous_suffix(base_label, threshold, feature_type, feature_metadata, include_units):
        """
        Add continuous feature suffix.

        Parameters
        ----------
        base_label : str
            Base label
        threshold : float
            Threshold value
        feature_type : str
            Feature type
        feature_metadata : list
            Metadata array
        include_units : bool
            Include units flag

        Returns
        -------
        str
            Label with threshold suffix

        Examples
        --------
        >>> label = FeatureLabelBuilder._add_continuous_suffix(
        ...     'distances: ALA5-LEU10', 4.5, 'distances', metadata, True
        ... )
        >>> print(label)  # 'distances: ALA5-LEU10 ≤ 4.50 Å'
        """
        if threshold == 0.5:
            return base_label

        if include_units:
            type_meta = FeatureMetadataUtils.get_top_level_metadata(
                feature_type, feature_metadata
            )
            unit = type_meta.get("units", "") if type_meta else ""
            unit_str = f" {unit}" if unit else ""
            return f"{base_label} ≤ {threshold:.2f}{unit_str}"

        return f"{base_label} <= {threshold:.2f}"

    @staticmethod
    def get_reverse_edge_semantics(type_meta) -> bool:
        """
        Check if feature type uses reverse edge semantics.

        For features where "positive" is actually undesirable (e.g., contacts),
        edge colors and symbols are reversed: left=bad (red/✗), right=good (green/✓).

        Parameters
        ----------
        type_meta : dict
            Feature type metadata

        Returns
        -------
        bool
            True if edges should be reversed

        Examples
        --------
        >>> # Contacts: Contact (1) is bad, Non-Contact (0) is good
        >>> reverse = FeatureLabelBuilder.get_reverse_edge_semantics(contact_meta)
        >>> print(reverse)  # True

        >>> # Distances: Small distance is good, large is bad
        >>> reverse = FeatureLabelBuilder.get_reverse_edge_semantics(distance_meta)
        >>> print(reverse)  # False

        Notes
        -----
        - True: left edge (≤) gets negative color/symbol (red/✗)
        - False: left edge (≤) gets positive color/symbol (green/✓)
        - Used for both edge colors and edge symbols
        """
        if not type_meta:
            return False
        vis_metadata = type_meta.get('visualization', {})
        return vis_metadata.get('reverse_edge_semantics', False)

    @staticmethod
    def _should_hide_prefix(
        feature_type: str,
        feature_metadata,
        hide_feature_type_prefix: bool
    ) -> bool:
        """
        Check if feature type prefix should be hidden.

        Parameters
        ----------
        feature_type : str
            Feature type
        feature_metadata : list
            Feature metadata array
        hide_feature_type_prefix : bool
            User setting

        Returns
        -------
        bool
            True if prefix should be hidden

        Examples
        --------
        >>> # User wants to hide, metadata allows
        >>> should_hide = FeatureLabelBuilder._should_hide_prefix(
        ...     'torsions', metadata, True
        ... )
        >>> print(should_hide)  # True

        >>> # User wants to hide, metadata does NOT allow
        >>> should_hide = FeatureLabelBuilder._should_hide_prefix(
        ...     'distances', metadata, True
        ... )
        >>> print(should_hide)  # False
        """
        if not hide_feature_type_prefix or feature_metadata is None:
            return False

        type_metadata = FeatureMetadataUtils.get_top_level_metadata(
            feature_type, feature_metadata
        )
        if type_metadata:
            vis_metadata = type_metadata.get('visualization', {})
            allow_hide = vis_metadata.get('allow_hide_prefix', False)
            return allow_hide

        return False
