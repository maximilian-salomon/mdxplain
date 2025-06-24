# mdxplain - A Python toolkit for molecular dynamics trajectory analysis
# distances - Distance feature type for molecular dynamics analysis
#
# Distance feature type implementation with pairwise distance calculations.
#
# Author: Maximilian Salomon
# Created with assistance from Claude-4-Sonnet and Cursor AI.
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

from typing import List, Optional

from ..interfaces.feature_type_base import FeatureTypeBase
from .distance_calculator import DistanceCalculator
from .reduce_distance_metrics import ReduceDistanceMetrics


class Distances(FeatureTypeBase):
    """Distance feature type for molecular dynamics analysis."""

    ReduceMetrics = ReduceDistanceMetrics
    """Available reduce metrics for distance features."""

    def __init__(self, ref: Optional[str] = None):
        """
        Initialize distance calculator for molecular dynamics analysis.

        Args:
            ref: Reference structure for distance calculations
        """
        super().__init__()
        self.ref = ref

    def init_calculator(self, use_memmap=False, cache_path=None, chunk_size=None):
        self.calculator = DistanceCalculator(
            use_memmap=use_memmap, cache_path=cache_path, chunk_size=chunk_size
        )

    def compute(self, input_data=None, feature_names=None):
        """Compute distances for given trajectories."""
        if self.calculator is None:
            raise ValueError("Calculator not initialized. Call init_calculator() first.")
        return self.calculator.compute(trajectories=input_data, ref=self.ref)

    def get_dependencies(self) -> List[str]:
        """
        Get list of dependencies required for distance calculations.

        Returns:
            Empty list as distances have no dependencies
        """
        return []

    @staticmethod
    def __str__() -> str:
        """Return string representation of the distance feature type."""
        return "distances"
