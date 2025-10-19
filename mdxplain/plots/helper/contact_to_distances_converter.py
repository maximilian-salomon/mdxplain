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
Helper class for converting contact-based feature selectors to distance-based.

Contact features are boolean (0/1) indicating presence/absence of contacts,
which are not suitable for violin plots. This helper creates equivalent
distance-based feature selectors for the same atom pairs, enabling
continuous value visualization.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Tuple, Optional

if TYPE_CHECKING:
    from ...pipeline.entities.pipeline_data import PipelineData

from ...feature_selection.managers.feature_selector_manager import FeatureSelectorManager
from ...utils.output_utils import OutputUtils


class ContactToDistancesConverter:
    """
    Helper class for converting contact features to distances.

    Contacts are boolean features (0/1) that indicate whether atoms are
    within a cutoff distance. These are not suitable for violin plots
    which require continuous values. This helper creates a new feature
    selector with distances for the same atom pairs.

    The conversion preserves:
    - Exact same atom pairs
    - Same feature ordering
    - Same trajectory coverage

    Examples
    --------
    >>> # Ensure feature selector uses continuous values
    >>> continuous_selector = ContactToDistancesConverter.convert_contacts_to_distances(
    ...     pipeline_data, "important_contacts"
    ... )
    >>> # Returns "important_contacts_distances" if conversion needed
    >>> # Returns "important_contacts" if already continuous
    """

    @staticmethod
    def convert_contacts_to_distances(
        pipeline_data: PipelineData,
        feature_selector_name: str
    ) -> Tuple[str, bool]:
        """
        Ensure feature selector uses continuous distance features.

        If the selector contains 'contacts' (boolean features), creates
        a new selector '{name}_distances' with distances for the same
        atom pairs. If selector already uses continuous features, returns
        original name unchanged.

        Parameters
        ----------
        pipeline_data : PipelineData
            Pipeline data container with feature selectors
        feature_selector_name : str
            Name of the feature selector to check/convert

        Returns
        -------
        Tuple[str, bool, Optional[float]]
            Tuple of (selector_name, is_temporary, contact_cutoff)
            - selector_name: Name of continuous feature selector
            - is_temporary: True if temporary selector was created, False otherwise
            - contact_cutoff: Contact cutoff value if converted from contacts, None otherwise

        Raises
        ------
        ValueError
            If feature selector not found or conversion fails

        Examples
        --------
        >>> # Selector with contacts - creates new distances selector
        >>> new_name = ContactToDistancesConverter.convert_contacts_to_distances(
        ...     pipeline_data, "key_contacts"
        ... )
        >>> print(new_name)  # "key_contacts_distances"

        >>> # Selector already with distances - returns original
        >>> name = ContactToDistancesConverter.convert_contacts_to_distances(
        ...     pipeline_data, "key_distances"
        ... )
        >>> print(name)  # "key_distances"

        Notes
        -----
        Boolean contact features cannot be meaningfully visualized in
        violin plots. This method automatically creates a distances-based
        version for visualization while preserving the original selector
        for analysis purposes.

        The new selector name follows the pattern: "{original}_distances"
        """
        if feature_selector_name not in pipeline_data.selected_feature_data:
            available = list(pipeline_data.selected_feature_data.keys())
            raise ValueError(
                f"Feature selector '{feature_selector_name}' not found. "
                f"Available: {available}"
            )

        selector_data = pipeline_data.selected_feature_data[feature_selector_name]

        # Check if selector contains contacts
        if "contacts" not in selector_data.selections:
            # Already continuous - return original name, not temporary, no cutoff
            return feature_selector_name, False, None

        # Need to create distances version
        new_selector_name = f"{feature_selector_name}_distances"

        # Check if distances version already exists
        if new_selector_name in pipeline_data.selected_feature_data:
            # Already exists, not temporary (created elsewhere), extract cutoff
            contact_cutoff = ContactToDistancesConverter._extract_contact_cutoff(
                pipeline_data
            )
            return new_selector_name, False, contact_cutoff

        # Check if distances feature exists
        if "distances" not in pipeline_data.feature_data:
            raise ValueError(
                f"Cannot convert contacts to distances: 'distances' feature not computed. "
                "Please compute distances first using pipeline.feature.add_feature(Distances(), ...)"
            )

        # Create new selector with distances (suppress all print output)
        manager = FeatureSelectorManager()

        with OutputUtils.suppress_output():
            manager.create(pipeline_data, new_selector_name)

            # Copy all feature selections (convert contacts to distances, keep rest)
            ContactToDistancesConverter._copy_selections_to_new_selector(
                pipeline_data, selector_data, new_selector_name, manager
            )

            # Execute selection with same reference trajectory
            reference_traj = selector_data.reference_trajectory
            manager.select(pipeline_data, new_selector_name, reference_traj=reference_traj)

        # Extract contact cutoff for threshold visualization
        contact_cutoff = ContactToDistancesConverter._extract_contact_cutoff(
            pipeline_data
        )

        # Return selector name, temporary flag, and cutoff
        return new_selector_name, True, contact_cutoff

    @staticmethod
    def _copy_selections_to_new_selector(
        pipeline_data,
        selector_data,
        new_selector_name: str,
        manager
    ) -> None:
        """
        Copy selections from original selector to new selector.

        Parameters
        ----------
        pipeline_data : PipelineData
            Pipeline data container
        selector_data : SelectedFeatureData
            Original selector data
        new_selector_name : str
            Name of new selector
        manager : FeatureSelectorManager
            Manager for adding selections

        Returns
        -------
        None
            Adds selections to new selector via manager
        """
        for feature_key, selections_list in selector_data.selections.items():
            # Convert contacts to distances, keep rest unchanged
            target_feature_key = "distances" if feature_key == "contacts" else feature_key

            # Copy all selections for this feature
            for selection_dict in selections_list:
                manager.add_selection(
                    pipeline_data,
                    new_selector_name,
                    target_feature_key,
                    selection_dict["selection"],
                    use_reduced=selection_dict["use_reduced"],
                    common_denominator=selection_dict.get("common_denominator", True),
                    traj_selection=selection_dict.get("traj_selection", "all"),
                    require_all_partners=selection_dict.get("require_all_partners", False)
                )

    @staticmethod
    def _extract_contact_cutoff(pipeline_data: PipelineData) -> Optional[float]:
        """
        Extract contact cutoff from contacts metadata.

        Retrieves the cutoff value used when computing contact features.
        All trajectories use the same cutoff (scientific best practice).

        Parameters
        ----------
        pipeline_data : PipelineData
            Pipeline data container

        Returns
        -------
        Optional[float]
            Contact cutoff value in Angstroms, or None if not found

        Examples
        --------
        >>> cutoff = ContactToDistancesConverter._extract_contact_cutoff(
        ...     pipeline_data
        ... )
        >>> print(cutoff)  # 4.5

        Notes
        -----
        Extracts from first trajectory's contacts metadata. Assumes all
        trajectories use the same cutoff (which is scientifically correct).
        """
        if "contacts" not in pipeline_data.feature_data:
            return None

        # Get cutoff from contacts metadata (first trajectory)
        contacts_feature_data = pipeline_data.feature_data["contacts"]
        first_traj_idx = next(iter(contacts_feature_data.keys()))
        contacts_metadata = contacts_feature_data[first_traj_idx].feature_metadata

        # Extract cutoff from computation_params
        computation_params = contacts_metadata.get("computation_params", {})
        return computation_params.get("cutoff")

    @staticmethod
    def cleanup_temporary_selector(
        pipeline_data: PipelineData, selector_name: str
    ) -> None:
        """
        Remove temporary distance selector.

        Silently removes a feature selector that was created temporarily
        for visualization purposes.

        Parameters
        ----------
        pipeline_data : PipelineData
            Pipeline data container
        selector_name : str
            Name of selector to remove

        Returns
        -------
        None
            Removes selector from pipeline_data

        Examples
        --------
        >>> # After creating temporary selector
        >>> selector, is_temp = ContactToDistancesConverter.convert_contacts_to_distances(
        ...     pipeline_data, "contacts_only"
        ... )
        >>> # Use selector...
        >>> if is_temp:
        ...     ContactToDistancesConverter.cleanup_temporary_selector(
        ...         pipeline_data, selector
        ...     )

        Notes
        -----
        Only removes selectors that actually exist in pipeline_data.
        Silently does nothing if selector doesn't exist.
        """
        if selector_name not in pipeline_data.selected_feature_data:
            return

        manager = FeatureSelectorManager()
        with OutputUtils.suppress_output():
            manager.remove_selector(pipeline_data, selector_name)
