# mdxplain - A Python toolkit for molecular dynamics trajectory analysis
# contacts - Contact feature type for molecular dynamics analysis
#
# Contact feature type implementation with distance-based contact detection.
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

from ..distances.distances import Distances
from .contact_calculator import ContactCalculator
from .reduce_contact_metrics import ReduceContactMetrics
from ..interfaces.feature_type_base import FeatureTypeBase
from typing import List

class Contacts(FeatureTypeBase):
    """Contact feature type for molecular dynamics analysis."""
    
    ReduceMetrics = ReduceContactMetrics()
    """Available reduce metrics for contact features."""

    def __init__(self, cutoff=4.5):
        """
        Initialize contact calculator for molecular dynamics analysis.
        
        Args:
            cutoff: Distance cutoff for contact determination (in Angstrom)
        """
        super().__init__()
        self.cutoff = cutoff

    def init_calculator(self, use_memmap=False, cache_path=None, 
                        chunk_size=None):
        """Initialize the contact calculator with given parameters."""
        self.calculator = ContactCalculator(use_memmap=use_memmap, 
                                                cache_path=cache_path, 
                                                chunk_size=chunk_size
        )

    def compute(self, input_data, feature_names):
        """Compute contacts from distance data."""
        if self.calculator is None:
            raise ValueError("Calculator not initialized. Call init_calculator() first.")
        return self.calculator.compute(distances=input_data, cutoff=self.cutoff), feature_names

    def get_dependencies(self) -> List[str]:
        """
        Get list of dependencies required for contact calculations.
        
        Returns:
            List containing Distances dependency
        """
        return [str(Distances)] 
    
    @staticmethod
    def __str__() -> str:
        """Return string representation of the contact feature type."""
        return "contacts"

    def get_input(self):
        """
        Get the input feature type.
        """
        return str(Distances)