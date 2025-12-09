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

"""Integration tests for ContactToDistancesConverter."""

import numpy as np
import pytest
import mdtraj as md
from mdxplain.pipeline.manager.pipeline_manager import PipelineManager
from mdxplain.feature.feature_type.contacts.contacts import Contacts
from mdxplain.feature.feature_type.distances.distances import Distances
from mdxplain.plots.helper.contact_to_distances_converter import ContactToDistancesConverter


CONTACT_CUTOFF = 4.5
POSITIONS_NM = [0.0, 0.3, 0.6, 1.0, 1.5]  # yields three contacts below 4.5Å
N_FRAMES = 5


def build_test_pipeline(
    include_distances: bool = True,
    include_contacts: bool = True,
    reduce_contacts: bool = False,
    reduction_threshold: float = 0.8,
    cutoff: float = CONTACT_CUTOFF,
    variation_per_frame: float = 0.01,
) -> PipelineManager:
    """
    Create a deterministic pipeline with optional contact reduction.

    Parameters
    ----------
    include_distances : bool, optional
        Whether to compute distance features, by default True.
    include_contacts : bool, optional
        Whether to compute contact features, by default True.
    reduce_contacts : bool, optional
        Whether to reduce contacts, by default False.
    reduction_threshold : float, optional
        Minimum contact frequency for reduction, by default 0.8.
    cutoff : float, optional
        Contact cutoff in Angstrom, by default CONTACT_CUTOFF.
    variation_per_frame : float, optional
        Additional offset per frame and residue index to create small
        distance fluctuations, by default 0.01 nm.

    Returns
    -------
    PipelineManager
        A pipeline with synthetic trajectory, labels, and requested features.
    """
    topology = md.Topology()
    chain = topology.add_chain()
    residues = [topology.add_residue(name, chain) for name in ["ALA", "GLY", "VAL", "ALA", "GLY"]]

    for residue in residues:
        topology.add_atom("CA", md.element.carbon, residue)

    coordinates = []
    for frame in range(N_FRAMES):
        frame_coords = []
        for atom_idx, pos in enumerate(POSITIONS_NM):
            frame_coords.append([pos + variation_per_frame * frame * atom_idx, 0.0, 0.0])
        coordinates.append(frame_coords)
    traj = md.Trajectory(np.array(coordinates), topology)

    pipeline = PipelineManager()
    pipeline.data.trajectory_data.trajectories = [traj]
    pipeline.data.trajectory_data.trajectory_names = ["test"]

    residue_metadata = []
    for res in traj.topology.residues:
        residue_metadata.append({
            "resid": res.resSeq + 1,
            "seqid": res.index + 1,
            "resname": res.name,
            "aaa_code": res.name,
            "a_code": res.name[0],
            "consensus": None,
            "full_name": f"{res.name}{res.index + 1}",
            "index": res.index
        })

    pipeline.data.trajectory_data.res_label_data = {0: residue_metadata}

    if include_distances:
        pipeline.feature.add_feature(Distances(excluded_neighbors=0))

    if include_contacts:
        pipeline.feature.add_feature(Contacts(cutoff=cutoff))

    if reduce_contacts:
        pipeline.feature.reduce.contacts.frequency(threshold_min=reduction_threshold)

    return pipeline


class TestContactToDistanceConversionReduced:
    """Test conversion when contacts are reduced but distances are not."""

    def setup_method(self):
        """Create pipeline with reduced contacts and selected pairs."""
        self.pipeline = build_test_pipeline(reduce_contacts=True)
        self.kept_indices = self.pipeline.data.feature_data["contacts"][0].reduction_info["kept_indices"]

        self.pipeline.feature_selector.create("contacts_only")
        self.pipeline.feature_selector.add_selection("contacts_only", "contacts", "all", use_reduced=True)
        self.pipeline.feature_selector.select("contacts_only")

    def test_converter_creates_distances_selector(self):
        """Test that converter creates new distances selector."""
        name, is_temp, cutoff = ContactToDistancesConverter.convert_contacts_to_distances(
            self.pipeline.data, "contacts_only"
        )

        assert name == "contacts_only_distances"
        assert is_temp is True
        assert cutoff == 4.5
        assert "contacts_only_distances" in self.pipeline.data.selected_feature_data

    def test_converted_selector_has_correct_indices(self):
        """Test that converted selector uses correct atom pairs."""
        # Get kept_indices from contacts reduction
        kept_indices = self.pipeline.data.feature_data["contacts"][0].reduction_info["kept_indices"]

        # Call converter
        name, _, _ = ContactToDistancesConverter.convert_contacts_to_distances(
            self.pipeline.data, "contacts_only"
        )

        # Get selection results for distances
        distances_selector = self.pipeline.data.selected_feature_data[name]
        distances_indices = distances_selector.selection_results["distances"]["trajectory_indices"][0]["indices"]

        # Indices should match kept_indices from contacts
        assert distances_indices == kept_indices.tolist()

    def test_get_selected_data_returns_same_features(self):
        """Test that both selectors target the same atom pairs."""
        name, _, cutoff = ContactToDistancesConverter.convert_contacts_to_distances(
            self.pipeline.data, "contacts_only"
        )

        contacts_data = self.pipeline.data.get_selected_data("contacts_only")
        distances_data = self.pipeline.data.get_selected_data(name)

        expected_distances = self.pipeline.data.feature_data["distances"][0].data[:, self.kept_indices]

        assert np.allclose(distances_data, expected_distances)
        assert np.array_equal(contacts_data.astype(bool), distances_data <= cutoff)
        # Distances should be taken from original data, not reduced
        distances_feature = self.pipeline.data.feature_data["distances"][0]
        assert distances_feature.reduced_data is None
        assert distances_feature.reduction_info is None

    def test_ignores_reduced_distances(self):
        """Distances reduction must not affect mapped indices or data."""
        # Reduce distances to a different (empty) set to ensure converter cannot rely on reduced data
        self.pipeline.feature.reduce.distances.cv(threshold_min=0.45)
        distances_feature = self.pipeline.data.feature_data["distances"][0]
        assert distances_feature.reduction_info is not None  # reduction exists

        name, _, _ = ContactToDistancesConverter.convert_contacts_to_distances(
            self.pipeline.data, "contacts_only"
        )

        distances_selector = self.pipeline.data.selected_feature_data[name]
        traj_data = distances_selector.selection_results["distances"]["trajectory_indices"][0]

        assert traj_data["indices"] == self.kept_indices.tolist()
        assert all(flag is False for flag in traj_data["use_reduced"])

        expected = distances_feature.data[:, self.kept_indices]
        converted = self.pipeline.data.get_selected_data(name)
        assert np.allclose(converted, expected)

    def test_cleanup_removes_temporary_selector(self):
        """Test that cleanup removes temporary selector."""
        # Call converter
        name, is_temp, _ = ContactToDistancesConverter.convert_contacts_to_distances(
            self.pipeline.data, "contacts_only"
        )

        # Selector should exist
        assert name in self.pipeline.data.selected_feature_data
        assert is_temp is True

        # Cleanup
        ContactToDistancesConverter.cleanup_temporary_selector(self.pipeline.data, name)

        # Selector should not exist anymore
        assert name not in self.pipeline.data.selected_feature_data


class TestContactToDistanceConversionAlreadyExists:
    """Test conversion when distances selector already exists."""

    def setup_method(self):
        """Create synthetic trajectory with existing distances selector."""
        self.pipeline = build_test_pipeline(reduce_contacts=True)

        self.pipeline.feature_selector.create("contacts_only")
        self.pipeline.feature_selector.add_selection("contacts_only", "contacts", "all", use_reduced=True)
        self.pipeline.feature_selector.select("contacts_only")

        # MANUALLY create distances selector
        self.pipeline.feature_selector.create("contacts_only_distances")
        self.pipeline.feature_selector.add_selection("contacts_only_distances", "distances", "all")
        self.pipeline.feature_selector.select("contacts_only_distances")

    def test_converter_returns_existing_selector(self):
        """Test that converter returns existing selector without recreation."""
        name, is_temp, cutoff = ContactToDistancesConverter.convert_contacts_to_distances(
            self.pipeline.data, "contacts_only"
        )

        assert name == "contacts_only_distances"
        assert is_temp is False  # Not temporary because it existed before
        assert cutoff == 4.5


class TestContactToDistanceConversionAlreadyDistances:
    """Test conversion when selector already uses distances."""

    def setup_method(self):
        """Create synthetic trajectory with distances selector."""
        self.pipeline = build_test_pipeline(reduce_contacts=False)

        self.pipeline.feature_selector.create("distances_only")
        self.pipeline.feature_selector.add_selection("distances_only", "distances", "all")
        self.pipeline.feature_selector.select("distances_only")

    def test_converter_returns_original_selector(self):
        """Test that converter returns original when already using distances."""
        name, is_temp, cutoff = ContactToDistancesConverter.convert_contacts_to_distances(
            self.pipeline.data, "distances_only"
        )

        assert name == "distances_only"  # Unchanged
        assert is_temp is False
        assert cutoff is None


class TestContactToDistanceConversionErrors:
    """Test error cases in conversion."""

    def setup_method(self):
        """Create synthetic trajectory without distances feature."""
        # Compute contacts first (requires distances), then drop distances to emulate missing dependency
        self.pipeline = build_test_pipeline(reduce_contacts=True)
        self.pipeline.data.feature_data.pop("distances")

        self.pipeline.feature_selector.create("contacts_only")
        self.pipeline.feature_selector.add_selection("contacts_only", "contacts", "all", use_reduced=True)
        self.pipeline.feature_selector.select("contacts_only")

    def test_converter_raises_when_distances_not_computed(self):
        """Test that converter raises error when distances feature missing."""
        with pytest.raises(ValueError) as exc_info:
            ContactToDistancesConverter.convert_contacts_to_distances(
                self.pipeline.data, "contacts_only"
            )

        assert "Please compute distances first" in str(exc_info.value)


class TestContactToDistanceConversionSelectorReduction:
    """Test conversion when selector-level contact reduction is used."""

    def setup_method(self):
        """Create pipeline and reduce contacts via selector service."""
        self.pipeline = build_test_pipeline(reduce_contacts=False)

        self.pipeline.feature_selector.create("contacts_reduced")
        self.pipeline.feature_selector.add.contacts.with_frequency_reduction(
            "contacts_reduced",
            "all",
            threshold_min=0.8,
            cross_trajectory=True,
            use_reduced=False,
        )
        self.pipeline.feature_selector.select("contacts_reduced")

        self.original_indices = self.pipeline.data.selected_feature_data["contacts_reduced"].selection_results[
            "contacts"
        ]["trajectory_indices"][0]["indices"]

    def test_conversion_preserves_selector_reduction_indices(self):
        """Converted selector should mirror reduced contact indices."""
        name, is_temp, cutoff = ContactToDistancesConverter.convert_contacts_to_distances(
            self.pipeline.data, "contacts_reduced"
        )

        assert is_temp is True
        assert cutoff == CONTACT_CUTOFF

        distances_selector = self.pipeline.data.selected_feature_data[name]
        traj_data = distances_selector.selection_results["distances"]["trajectory_indices"][0]

        assert traj_data["indices"] == self.original_indices
        assert all(flag is False for flag in traj_data["use_reduced"])

        # Data matches original distances for those indices
        distances_feature = self.pipeline.data.feature_data["distances"][0].data
        expected = distances_feature[:, self.original_indices]
        converted = self.pipeline.data.get_selected_data(name)
        assert np.allclose(converted, expected)
