# mdxplain - A Python toolkit for molecular dynamics trajectory analysis
#
# Author: Maximilian Salomon
# Created with assistance from Claude Code (Claude Opus 4.8).
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

"""Integration tests for the structure_visualization PDB-writing services.

This module had been entirely untested, which let a regression ship: removing
the ``chunk_size`` attribute from ``StructureVisualizationManager`` broke
``create_pdb_with_beta_factors`` at runtime (the feature-importance service reads
``self._manager.chunk_size``) while every other suite stayed green, because none
of them touches this module. These tests drive both public PDB entry points end
to end on a real in-memory trajectory so that class of regression fails here
instead of in a user's production run.
"""

import glob
import os

import mdtraj as md
import numpy as np

from mdxplain.pipeline.manager.pipeline_manager import PipelineManager
from mdxplain.feature.feature_type.distances.distances import Distances
from mdxplain.comparison.entities.comparison_data import ComparisonData
from mdxplain.feature_importance.entities.feature_importance_data import (
    FeatureImportanceData,
)

RESIDUE_NAMES = ["ALA", "GLY", "VAL", "ALA", "GLY"]
N_FRAMES = 30


def _build_trajectory():
    """Build a real 5-residue, one-CA-per-residue trajectory over 30 frames.

    A real ``mdtraj.Trajectory`` is required because the PDB writer calls
    ``frame.save_pdb``; the mock trajectories used elsewhere lack it. Each atom
    drifts linearly with the frame index so the per-frame feature vectors differ
    and a centroid frame is well defined.
    """
    topology = md.Topology()
    chain = topology.add_chain()
    residues = [topology.add_residue(name, chain) for name in RESIDUE_NAMES]
    for residue in residues:
        topology.add_atom("CA", md.element.carbon, residue)

    coords = np.zeros((N_FRAMES, len(RESIDUE_NAMES), 3))
    for frame in range(N_FRAMES):
        for atom_idx in range(len(RESIDUE_NAMES)):
            coords[frame, atom_idx] = [atom_idx * 1.0, frame * 0.1, 0.0]
    return md.Trajectory(coords, topology)


def _residue_metadata(trajectory):
    """Build per-residue label metadata with a positional ``index`` per residue.

    The beta-factor path reads each residue's ``index`` from the selection
    metadata, so the labels must carry the same positional indices the distance
    pairs use.
    """
    metadata = []
    for res in trajectory.topology.residues:
        metadata.append(
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
        )
    return metadata


def _base_pipeline():
    """Build a pipeline with a real trajectory, distances, a selector and a cluster.

    This is the shared fixture for both PDB entry points: a processed feature
    selector ``feats`` over the distance pairs and a data selector ``cluster_0``
    covering every frame, from which a centroid frame can be taken.
    """
    trajectory = _build_trajectory()
    pipeline = PipelineManager()
    pipeline.data.trajectory_data.trajectories = [trajectory]
    pipeline.data.trajectory_data.trajectory_names = ["synthetic"]
    pipeline.data.trajectory_data.res_label_data = {0: _residue_metadata(trajectory)}

    pipeline.feature.add_feature(Distances(excluded_neighbors=0))

    pipeline.feature_selector.create("feats")
    pipeline.feature_selector.add_selection(
        "feats", "distances", "all", use_reduced=False
    )
    pipeline.feature_selector.select("feats")

    pipeline.data_selector.create("cluster_0")
    pipeline.data_selector.select_by_indices(
        "cluster_0", {0: list(range(N_FRAMES))}
    )
    return pipeline


def _written_pdbs(pipeline):
    """Return the PDB files the structure-viz output directory now holds."""
    output_dir = pipeline._structure_visualization_manager.output_dir
    return sorted(glob.glob(os.path.join(output_dir, "*.pdb")))


def test_create_pdb_with_beta_factors_reads_manager_config():
    """Beta-factor PDB creation runs end to end and reads the manager's config.

    This is the regression guard for the removed ``chunk_size``: the
    feature-importance service passes ``self._manager.use_memmap`` and
    ``self._manager.chunk_size`` into ``get_representative_frame``. If either
    attribute is missing the call raises ``AttributeError`` before any PDB is
    written, so exercising the full method — in ``centroid`` mode, which needs no
    fitted model — pins that contract. The test asserts the visualization entry
    is stored and a non-empty PDB lands on disk.
    """
    pipeline = _base_pipeline()
    n_pairs = pipeline.data.feature_data["distances"][0].data.shape[1]

    comp_data = ComparisonData("cmp", "one_vs_rest", "feats", ["cluster_0"])
    comp_data.add_sub_comparison(
        "cluster_0_vs_rest", ["cluster_0"], ["cluster_0"], (0, 1)
    )
    pipeline.data.comparison_data["cmp"] = comp_data

    fi_data = FeatureImportanceData("fi")
    fi_data.comparison_name = "cmp"
    fi_data.feature_selector = "feats"
    fi_data.add_comparison_result(
        np.linspace(1.0, 0.1, n_pairs),
        {"comparison": "cluster_0_vs_rest", "labels": (0, 1)},
    )
    pipeline.data.feature_importance_data["fi"] = fi_data

    pipeline.structure_visualization.feature_importance.create_pdb_with_beta_factors(
        "viz", "fi", n_top=3, representative_mode="centroid"
    )

    assert "viz" in pipeline.data.structure_visualization_data
    pdbs = _written_pdbs(pipeline)
    assert len(pdbs) == 1
    assert os.path.getsize(pdbs[0]) > 0
    assert md.load(pdbs[0]).topology.n_atoms == len(RESIDUE_NAMES)


def test_create_representative_pdbs_from_centroids():
    """The feature path writes one centroid PDB per data selector.

    ``create_representative_pdbs`` needs neither comparison nor
    feature-importance data — only a centroid feature selector and data
    selectors — so it covers the second public PDB entry point on the lightest
    possible fixture. The test asserts a PDB is written and is a valid structure.
    """
    pipeline = _base_pipeline()

    pipeline.structure_visualization.feature.create_representative_pdbs(
        "viz_feat", ["cluster_0"], selector_centroid="feats", selector_features=None
    )

    assert "viz_feat" in pipeline.data.structure_visualization_data
    pdbs = _written_pdbs(pipeline)
    assert len(pdbs) == 1
    assert md.load(pdbs[0]).topology.n_atoms == len(RESIDUE_NAMES)
