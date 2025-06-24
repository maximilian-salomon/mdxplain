# mdxplain - A Python toolkit for molecular dynamics trajectory analysis
# feature_type_base - Base interface for feature types
#
# Abstract base class defining the interface for all feature types in mdxplain.
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

from abc import ABC, abstractmethod
from typing import List, Tuple
import numpy as np

class FeatureTypeBase(ABC):
    """Base class for all feature types."""
    
    def __init__(self):
        """Initialize the feature type with a calculator."""
        self.calculator = None
    
    @abstractmethod
    def get_dependencies(self) -> List['FeatureTypeBase']:
        """
        Get list of dependencies required for this feature type.
        
        Returns:
            List of feature type objects that must be computed first
        """
        pass
    
    @staticmethod
    @abstractmethod
    def __str__() -> str:
        """
        Return string representation of the feature type.
        Used as key for storing in feature dictionaries.
        
        Returns:
            String identifier for this feature type
        """
        pass 

    @abstractmethod
    def init_calculator(self, **kwargs):
        """
        Initialize the calculator for this feature type.
        
        Args:
            **kwargs: Parameters for calculator initialization
        """
        pass

    @abstractmethod
    def compute(self, input_data=None, feature_names=None) -> Tuple[np.ndarray, List[str]]:
        """
        Compute the feature type.
        
        Args:
            **kwargs: Parameters for computation
        """
        pass
    
    def get_input(self):
        """
        Get the input feature type.
        """
        return None
