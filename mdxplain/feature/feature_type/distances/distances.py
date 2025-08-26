# mdxplain - A Python toolkit for molecular dynamics trajectory analysis
#
# Author: Maximilian Salomon
# Created with assistance from Cursor IDE (Claude Sonnet 4.0, occasional Claude Sonnet 3.7 and Gemini 2.5 Pro).
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
Distance feature type implementation for molecular dynamics analysis.

Distance feature type implementation with pairwise distance calculations
for analyzing molecular dynamics trajectories using MDTraj.
"""

from typing import List

from ..interfaces.feature_type_base import FeatureTypeBase
from .distance_calculator import DistanceCalculator
from .reduce_distance_metrics import ReduceDistanceMetrics


class Distances(FeatureTypeBase):
    """
    Distance feature type for calculating pairwise distances between atoms/residues.

    Computes all pairwise distances from molecular dynamics trajectories using
    MDTraj's distance calculation functions. The reference trajectory determines
    which atom pairs are computed and provides the basis for feature naming.
    This ensures consistent feature names when comparing different trajectories.

    This is a base feature type with no dependencies - other features like
    contacts use distance data as input.

    Examples:
    ---------
    >>> # Basic distance calculation via TrajectoryData
    >>> traj_data = TrajectoryData()
    >>> traj_data.load_trajectories('simulation/')
    >>> traj_data.add_feature(Distances())

    >>> # Distance calculation with memory mapping
    >>> distances = Distances()
    >>> traj_data.add_feature(distances, use_memmap=True, cache_path='./cache/')

    >>> # Distance calculation with reference trajectory for consistent naming
    >>> ref_traj = traj_data.trajectories[7]  # Use specific trajectory as reference
    >>> traj_data.add_feature(Distances(ref=ref_traj))
    """

    ReduceMetrics = ReduceDistanceMetrics
    """Available reduce metrics for distance features."""

    def __init__(self, ref=None, excluded_neighbors=1):
        """
        Initialize distance feature type with optional reference trajectory.

        Parameters:
        -----------
        ref : mdtraj.Trajectory, optional
            Reference trajectory that determines which atom pairs are computed and serves
            as basis for feature name generation. When comparing trajectories with different
            sequences (e.g., wildtype vs mutants), using a common reference ensures
            consistent feature naming across trajectories. Without reference, feature names
            depend on the specific trajectory topology, making cross-trajectory comparisons
            impossible. If None, uses first trajectory as reference.
        excluded_neighbors : int, default=0
            Number of nearest neighbors to consider for distance calculation.
            Chain Breaks are automatically excluded. Meassured by jump in the seqid of a residue.
            If 0, all pairs are computed.
            If 1, only nearest neighbors are computed.
            If 2, only nearest neighbors and their neighbors are computed.
            If 3, only nearest neighbors and their neighbors and their neighbors are computed.
            etc.

        Returns:
        --------
        None

        Examples:
        ---------
        >>> # Use first trajectory as reference (automatic)
        >>> distances = Distances()

        >>> # Use specific reference trajectory for consistent naming
        >>> traj_data = TrajectoryData()
        >>> traj_data.load_trajectories('wildtype/')
        >>> ref_traj = traj_data.trajectories[2]  # Second trajectory as reference
        >>> traj_data.add_feature(Distances(ref=ref_traj))

        >>> # Use custom diagonal offset
        >>> distances = Distances(excluded_neighbors=2)
        """
        super().__init__()
        self.ref = ref
        self.excluded_neighbors = excluded_neighbors

    def init_calculator(self, use_memmap=False, cache_path="./cache", chunk_size=10000):
        """
        Initialize the distance calculator with specified configuration.

        Parameters:
        -----------
        use_memmap : bool, default=False
            Whether to use memory mapping for large datasets
        cache_path : str, optional
            Directory path for storing cache files when using memory mapping
        chunk_size : int, optional
            Number of frames to process per chunk (None for automatic sizing)

        Returns:
        --------
        None

        Examples:
        ---------
        >>> # Basic initialization
        >>> distances.init_calculator()

        >>> # With memory mapping for large datasets
        >>> distances.init_calculator(use_memmap=True, cache_path='./cache/')

        >>> # With custom chunk size
        >>> distances.init_calculator(chunk_size=500)
        """
        self.calculator = DistanceCalculator(
            use_memmap=use_memmap, cache_path=cache_path, chunk_size=chunk_size
        )

    def compute(self, input_data, feature_metadata):
        """
        Compute pairwise distances from molecular dynamics trajectories.

        Parameters:
        -----------
        input_data : mdtraj.Trajectory or list
            MD trajectories to compute distances from
        feature_metadata : dict, optional
            Not used for distances (base feature type)

        Returns:
        --------
        tuple[numpy.ndarray, dict]
            Tuple containing (distance_matrix, feature_metadata) where distance_matrix
            is shape (n_frames, n_pairs) and feature_metadata is structured metadata
            with features in same order as data columns

        Examples:
        ---------
        >>> # Compute distances from trajectories
        >>> distances = Distances()
        >>> distances.init_calculator()
        >>> data, names = distances.compute(input_data=trajectories)
        >>> print(f"Distance matrix shape: {data.shape}")
        >>> print(f"First few pairs: {names[:5]}")

        >>> # Using memory mapping for large datasets
        >>> distances.init_calculator(use_memmap=True, cache_path='./cache/')
        >>> data, names = distances.compute(input_data=large_trajectories)

        Raises:
        -------
        ValueError
            If trajectories have different numbers of residues
            If calculator is not initialized
        """
        if self.calculator is None:
            raise ValueError(
                "Calculator not initialized. Call init_calculator() first."
            )
        
        return self.calculator.compute(
            input_data=input_data,
            excluded_neighbors=self.excluded_neighbors,
            res_metadata=feature_metadata,
        )

    def get_dependencies(self) -> List[str]:
        """
        Get list of feature type dependencies for distance calculations.

        Parameters:
        -----------
        None

        Returns:
        --------
        List[str]
            Empty list as distances are a base feature with no dependencies

        Examples:
        ---------
        >>> distances = Distances()
        >>> print(distances.get_dependencies())
        []
        """
        return []

    @classmethod
    def get_type_name(cls):  # noqa: vulture
        """
        Get the type name as class method.

        Parameters:
        -----------
        None

        Returns:
        --------
        str
            String identifier 'distances'

        Examples:
        ---------
        >>> print(Distances.get_type_name())
        'distances'
        """
        return "distances"

    def get_input(self):
        """
        Get the input feature type that distances depend on.

        Parameters:
        -----------
        None

        Returns:
        --------
        None
            None since distances are a base feature with no input dependencies

        Examples:
        ---------
        >>> distances = Distances()
        >>> print(distances.get_input())
        None
        """
        return None
