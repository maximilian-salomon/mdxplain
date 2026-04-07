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

"""Integration tests for get_selected_data mapping build/caching behavior."""

import numpy as np
import mdtraj as md
from unittest.mock import patch

from mdxplain.pipeline.manager.pipeline_manager import PipelineManager
from mdxplain.pipeline.helper.selection_matrix_helper import SelectionMatrixHelper
from mdxplain.feature.feature_type.distances.distances import Distances


class TestGetSelectedDataMappingCache:
    """Validate matrix and mapping correctness across lazy mapping cache path."""

    def setup_method(self):
        """Create deterministic single-trajectory setup with distance features."""
        topology = md.Topology()
        chain = topology.add_chain()
        residue_names = ["ALA", "GLY", "VAL", "SER"]
        residues = [topology.add_residue(name, chain) for name in residue_names]
        for residue in residues:
            topology.add_atom("CA", md.element.carbon, residue)

        coordinates = []
        for frame in range(20):
            frame_coords = []
            for atom_idx in range(4):
                frame_coords.append([atom_idx * 0.2, frame * 0.01, 0.0])
            coordinates.append(frame_coords)
        xyz = np.array(coordinates, dtype=np.float32)
        trajectory = md.Trajectory(xyz, topology)

        self.pipeline = PipelineManager(use_memmap=True, chunk_size=4, show_progress=False)
        self.pipeline.data.trajectory_data.trajectories = [trajectory]
        self.pipeline.data.trajectory_data.trajectory_names = ["traj0"]
        self.pipeline.data.trajectory_data.res_label_data = {
            0: [
                {
                    "resid": res.resSeq + 1,
                    "seqid": res.index + 1,
                    "resname": res.name,
                    "aaa_code": res.name,
                    "a_code": res.name[0],
                    "consensus": None,
                    "full_name": f"{res.name}{res.index + 1}",
                    "index": res.index,
                }
                for res in trajectory.topology.residues
            ]
        }

        self.pipeline.feature.add_feature(Distances(excluded_neighbors=0))
        self.original_data = self.pipeline.data.feature_data["distances"][0].data.copy()

        self.pipeline.feature_selector.create("all_dist")
        self.pipeline.feature_selector.add_selection("all_dist", "distances", "all")
        self.pipeline.feature_selector.select("all_dist", reference_traj=0)

        self.selected_frames = [2, 5, 9, 13, 17]
        self.pipeline.data_selector.create("subset")
        self.pipeline.data_selector.select_by_indices("subset", {0: self.selected_frames})
        self.expected_data = self.original_data[self.selected_frames, :]

    def test_lazy_frame_mapping_rebuild_from_cached_matrix(self):
        """Build matrix first without mapping, then request and validate mapping."""
        selected_data_no_map = self.pipeline.data.get_selected_data(
            "all_dist",
            data_selector="subset",
            return_frame_mapping=False,
        )
        np.testing.assert_array_almost_equal(selected_data_no_map, self.expected_data)

        cache_key = self.pipeline.data._get_matrix_cache_key("all_dist", "subset")
        assert cache_key in self.pipeline.data._matrix_cache
        _, cached_mapping = self.pipeline.data._matrix_cache[cache_key]
        assert cached_mapping is None

        with patch(
            "mdxplain.pipeline.helper.selection_matrix_helper.SelectionMatrixHelper._build_new_matrix",
            wraps=SelectionMatrixHelper._build_new_matrix,
        ) as build_new_matrix_mock, patch(
            "mdxplain.pipeline.helper.selection_matrix_helper.SelectionMatrixHelper._build_frame_mapping_only",
            wraps=SelectionMatrixHelper._build_frame_mapping_only,
        ) as build_mapping_only_mock:
            selected_data_with_map, frame_mapping = self.pipeline.data.get_selected_data(
                "all_dist",
                data_selector="subset",
                return_frame_mapping=True,
            )

        # Second call must hit cached matrix and only build mapping.
        assert build_new_matrix_mock.call_count == 0
        assert build_mapping_only_mock.call_count == 1
        np.testing.assert_array_almost_equal(selected_data_with_map, self.expected_data)

        expected_mapping = {
            row_idx: (0, frame_idx)
            for row_idx, frame_idx in enumerate(self.selected_frames)
        }
        assert frame_mapping == expected_mapping

        _, cached_mapping_after = self.pipeline.data._matrix_cache[cache_key]
        assert cached_mapping_after == expected_mapping
