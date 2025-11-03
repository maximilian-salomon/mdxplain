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
Residue importance calculator for feature importance visualization.

This module calculates importance scores for each residue based on
their occurrence in top important features, weighted by feature
importance scores.
"""

import numpy as np
import mdtraj as md
from typing import Dict, List, Any


class ResidueImportanceCalculator:
    """
    Calculator for residue importance scores.

    Computes importance scores for residues based on their occurrence
    in top important features, weighted by feature importance scores.

    Examples
    --------
    >>> # Calculate residue importance from top features
    >>> top_features = [
    ...     {"residue_indices": [5, 10], "importance_score": 0.85},
    ...     {"residue_indices": [5], "importance_score": 0.72}
    ... ]
    >>> importance = ResidueImportanceCalculator.calculate_residue_importance(
    ...     top_features
    ... )
    >>> print(importance[5])  # 1.57 (0.85 + 0.72)
    """

    @staticmethod
    def calculate_residue_importance(
        top_features: List[Dict[str, Any]]
    ) -> Dict[int, float]:
        """
        Calculate importance score for each residue.

        Computes weighted importance scores based on occurrence in
        top features, where each occurrence is weighted by the
        feature's importance score.

        Parameters
        ----------
        top_features : List[Dict[str, Any]]
            List of top feature dictionaries with keys:
            
            - "residue_indices": List[int]
            - "importance_score": float

        Returns
        -------
        Dict[int, float]
            Mapping from residue index to importance score

        Examples
        --------
        >>> top_features = [
        ...     {"residue_indices": [5, 10], "importance_score": 0.85},
        ...     {"residue_indices": [10], "importance_score": 0.60}
        ... ]
        >>> importance = ResidueImportanceCalculator.calculate_residue_importance(
        ...     top_features
        ... )
        >>> # Residue 5: 0.85, Residue 10: 0.85 + 0.60 = 1.45

        Notes
        -----
        - Each residue occurrence is weighted by feature importance
        - Residues in multiple features accumulate scores
        - Only residues in top features receive non-zero scores
        - Uses residue_indices from feature metadata (no string parsing)
        """
        residue_scores = {}

        for feature in top_features:
            # Extract residue indices from metadata (no string parsing!)
            residue_indices = feature.get("residue_indices", [])

            importance_score = feature.get("importance_score", 0.0)

            # Add weighted score to each involved residue
            for res_idx in residue_indices:
                if res_idx not in residue_scores:
                    residue_scores[res_idx] = 0.0
                residue_scores[res_idx] += importance_score

        return residue_scores

    @staticmethod
    def importance_to_beta_factors(
        residue_importance: Dict[int, float],
        topology: md.Topology
    ) -> np.ndarray:
        """
        Convert residue importance scores to beta factors.

        Normalizes importance scores to 0-1 range and expands
        to per-atom beta factors (all atoms in a residue get the
        same beta factor).

        Parameters
        ----------
        residue_importance : Dict[int, float]
            Mapping from residue index to importance score
        topology : mdtraj.Topology
            Topology object for atom-residue mapping

        Returns
        -------
        np.ndarray
            Array of beta factors for all atoms, shape (n_atoms,)

        Examples
        --------
        >>> residue_importance = {5: 1.57, 10: 1.45, 15: 0.92}
        >>> topology = md.load("structure.pdb").topology
        >>> beta_factors = ResidueImportanceCalculator.importance_to_beta_factors(
        ...     residue_importance, topology
        ... )
        >>> print(beta_factors.shape)  # (n_atoms,)
        >>> print(beta_factors.max())  # 1.0

        Notes
        -----
        - Scores are normalized to 0-1 range for nglview visualization
        - All atoms in a residue receive the same beta factor
        - Residues not in importance dict receive beta factor 0.0
        """
        normalized = ResidueImportanceCalculator._normalize_importance(
            residue_importance
        )
        return ResidueImportanceCalculator._expand_to_atoms(
            normalized, topology
        )

    @staticmethod
    def _normalize_importance(
        residue_importance: Dict[int, float]
    ) -> Dict[int, float]:
        """
        Normalize importance scores to 0-1 range.

        Parameters
        ----------
        residue_importance : Dict[int, float]
            Raw importance scores

        Returns
        -------
        Dict[int, float]
            Normalized scores (0-1)
        """
        if not residue_importance:
            return {}

        max_importance = max(residue_importance.values())
        if max_importance <= 0:
            return residue_importance

        return {
            res_idx: (score / max_importance)
            for res_idx, score in residue_importance.items()
        }

    @staticmethod
    def _expand_to_atoms(
        normalized_importance: Dict[int, float],
        topology: md.Topology
    ) -> np.ndarray:
        """
        Expand residue scores to per-atom beta factors.

        Parameters
        ----------
        normalized_importance : Dict[int, float]
            Normalized residue importance scores
        topology : mdtraj.Topology
            Topology for atom-residue mapping

        Returns
        -------
        np.ndarray
            Per-atom beta factors
        """
        beta_factors = np.zeros(topology.n_atoms, dtype=np.float32)

        for atom in topology.atoms:
            res_idx = atom.residue.index
            if res_idx in normalized_importance:
                beta_factors[atom.index] = normalized_importance[res_idx]

        return beta_factors

    @staticmethod
    def calculate_beta_factors_from_features(
        top_features: List[Dict[str, Any]],
        topology: md.Topology
    ) -> np.ndarray:
        """
        Calculate beta factors directly from top features.

        Convenience method that combines calculate_residue_importance
        and importance_to_beta_factors in one call.

        Parameters
        ----------
        top_features : List[Dict[str, Any]]
            List of top feature dictionaries
        topology : mdtraj.Topology
            Topology object

        Returns
        -------
        np.ndarray
            Array of beta factors for all atoms

        Examples
        --------
        >>> top_features = pipeline.feature_importance.get_top_features(
        ...     "dt_analysis", "cluster_0_vs_rest", n=10
        ... )
        >>> topology = pipeline.data.trajectory_data.get_topology()
        >>> beta_factors = ResidueImportanceCalculator.calculate_beta_factors_from_features(
        ...     top_features, topology
        ... )
        """
        # Calculate residue importance
        residue_importance = ResidueImportanceCalculator.calculate_residue_importance(
            top_features
        )

        # Convert to beta factors
        return ResidueImportanceCalculator.importance_to_beta_factors(
            residue_importance, topology
        )
