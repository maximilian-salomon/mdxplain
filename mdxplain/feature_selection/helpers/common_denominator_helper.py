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
Common denominator helper for feature selection system.

This module provides utilities for finding features that are present across
all trajectories using biological feature identity comparison.
"""

from typing import List


class CommonDenominatorHelper:
    """Helper class for common denominator operations in feature selection."""

    @staticmethod
    def apply_common_denominator(
        pipeline_data, feature_key: str, trajectory_results: dict
    ) -> dict:
        """
        Apply common denominator filtering to keep only features present in all trajectories.

        Uses biological feature identity (aaa_code, seqid, consensus) for comparison.

        Parameters:
        -----------
        pipeline_data : PipelineData
            Pipeline data object
        feature_key : str
            Feature key
        trajectory_results : dict
            Dictionary with trajectory-specific indices

        Returns:
        --------
        dict
            Filtered trajectory_results containing only common features
        """
        if len(trajectory_results) <= 1:
            # No filtering needed for single trajectory
            return trajectory_results

        # Extract feature identities for each trajectory
        trajectory_identities = {}
        for traj_idx, result in trajectory_results.items():
            identities = CommonDenominatorHelper.extract_feature_identities(
                pipeline_data, feature_key, traj_idx, result["indices"], result["use_reduced"]
            )
            trajectory_identities[traj_idx] = identities

        # Find common feature identities
        common_identities = CommonDenominatorHelper.find_common_feature_identities(trajectory_identities)

        # Filter indices to keep only common features using pre-computed identities
        filtered_results = {}
        for traj_idx, result in trajectory_results.items():
            traj_identities = trajectory_identities[traj_idx]
            filtered_indices, filtered_use_reduced = CommonDenominatorHelper.filter_using_identities(
                result["indices"], result["use_reduced"], 
                traj_identities, common_identities
            )
            if filtered_indices:  # Only keep trajectories with matching features
                filtered_results[traj_idx] = {
                    "indices": filtered_indices,
                    "use_reduced": filtered_use_reduced,
                }

        return filtered_results

    @staticmethod
    def filter_using_identities(
        indices: List[int], use_reduced_flags: List[bool],
        trajectory_identities: List[dict], common_identities: List[dict]
    ) -> tuple:
        """
        Filter indices using pre-computed identities.

        Parameters:
        -----------
        indices : List[int]
            Original indices
        use_reduced_flags : List[bool]
            Use reduced flags
        trajectory_identities : List[dict]
            Pre-computed feature identities for this trajectory
        common_identities : List[dict]
            Common feature identities to keep

        Returns:
        --------
        tuple
            Tuple of (filtered_indices, filtered_use_reduced)
        """
        filtered_indices = []
        filtered_use_reduced = []

        for identity in trajectory_identities:
            if CommonDenominatorHelper.identity_present_in_list(identity, common_identities):
                original_idx = identity["original_index"]
                filtered_indices.append(indices[original_idx])
                filtered_use_reduced.append(use_reduced_flags[original_idx])

        return filtered_indices, filtered_use_reduced

    @staticmethod
    def extract_feature_identities(
        pipeline_data, feature_key: str, traj_idx: int, 
        indices: List[int], use_reduced_flags: List[bool]
    ) -> List[dict]:
        """
        Extract biological feature identities for given indices.

        Parameters:
        -----------
        pipeline_data : PipelineData
            Pipeline data object
        feature_key : str
            Feature key
        traj_idx : int
            Trajectory index
        indices : List[int]
            Feature indices
        use_reduced_flags : List[bool]
            Whether each index uses reduced data

        Returns:
        --------
        List[dict]
            List of feature identities with biological information
        """
        identities = []
        feature_data = pipeline_data.feature_data[feature_key][traj_idx]

        for i, (idx, use_reduced) in enumerate(zip(indices, use_reduced_flags)):
            if use_reduced and not feature_data.reduced_feature_metadata:
                raise ValueError(
                    f"Reduced feature metadata not available for {feature_key} in trajectory {traj_idx}"
                )
            
            if use_reduced:
                metadata = feature_data.reduced_feature_metadata
            else:
                metadata = feature_data.feature_metadata

            features_list = metadata.get("features", [])
            # its a list of lists, where each sublist contains partners for the feature
            if idx < len(features_list) and len(features_list[idx]) > 0:
                # Extract identity from all partners of feature
                feature_partners = features_list[idx]
                if len(feature_partners) > 0:
                    partners_info = []
                    for partner in feature_partners:
                        residue = partner.get("residue", {})
                        partners_info.append({
                            "aaa_code": residue.get("aaa_code"),
                            "seqid": residue.get("seqid"),
                            "consensus": str(residue.get("consensus", "None"))
                        })
                    identity = {
                        "partners": partners_info,
                        "original_index": i
                    }
                    identities.append(identity)

        return identities

    @staticmethod
    def find_common_feature_identities(trajectory_identities: dict) -> List[dict]:
        """
        Find feature identities that are present in all trajectories.

        Parameters:
        -----------
        trajectory_identities : dict
            Dictionary mapping trajectory indices to their feature identities

        Returns:
        --------
        List[dict]
            List of feature identities common to all trajectories
        """
        if not trajectory_identities:
            return []

        # Start with first trajectory's identities
        first_traj = next(iter(trajectory_identities.keys()))
        common_identities = trajectory_identities[first_traj].copy()

        # Intersect with each other trajectory
        for traj_idx, identities in trajectory_identities.items():
            if traj_idx == first_traj:
                continue

            # Keep only identities present in current trajectory
            common_identities = [
                identity for identity in common_identities
                if CommonDenominatorHelper.identity_present_in_list(identity, identities)
            ]

        return common_identities

    @staticmethod
    def filter_to_common_features(
        pipeline_data, feature_key: str, traj_idx: int,
        indices: List[int], use_reduced_flags: List[bool], 
        common_identities: List[dict]
    ) -> tuple:
        """
        Filter indices to keep only those matching common identities.

        Parameters:
        -----------
        pipeline_data : PipelineData
            Pipeline data object
        feature_key : str
            Feature key
        traj_idx : int
            Trajectory index
        indices : List[int]
            Original indices
        use_reduced_flags : List[bool]
            Use reduced flags
        common_identities : List[dict]
            Common feature identities to keep

        Returns:
        --------
        tuple
            Tuple of (filtered_indices, filtered_use_reduced)
        """
        trajectory_identities = CommonDenominatorHelper.extract_feature_identities(
            pipeline_data, feature_key, traj_idx, indices, use_reduced_flags
        )

        filtered_indices = []
        filtered_use_reduced = []

        for identity in trajectory_identities:
            if CommonDenominatorHelper.identity_present_in_list(identity, common_identities):
                original_idx = identity["original_index"]
                filtered_indices.append(indices[original_idx])
                filtered_use_reduced.append(use_reduced_flags[original_idx])

        return filtered_indices, filtered_use_reduced

    @staticmethod
    def identity_present_in_list(target_identity: dict, identity_list: List[dict]) -> bool:
        """
        Check if target identity is present in identity list.

        Parameters:
        -----------
        target_identity : dict
            Identity to search for (with "partners" key containing all partner info)
        identity_list : List[dict]
            List of identities to search in

        Returns:
        --------
        bool
            True if identity is found (matches all partners' aaa_code, seqid, consensus)
        """
        target_partners = target_identity.get("partners", [])
        
        for identity in identity_list:
            identity_partners = identity.get("partners", [])
            
            # Check if all partners match
            if len(target_partners) == len(identity_partners):
                all_match = True
                for t_partner, i_partner in zip(target_partners, identity_partners):
                    if (t_partner.get("aaa_code") != i_partner.get("aaa_code") or
                        t_partner.get("seqid") != i_partner.get("seqid") or
                        # TODO: Not Sure if we want this. Consensus can be very specific. Maybe ignore it?
                        t_partner.get("consensus") != i_partner.get("consensus")):
                        all_match = False
                        break
                if all_match:
                    return True
        
        return False