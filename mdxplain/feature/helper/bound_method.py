# mdxplain - A Python toolkit for molecular dynamics trajectory analysis
#
# Author: Maximilian Salomon
# Created with assistance from Claude Code (Claude Sonnet 4.0).
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
Pickleable bound method implementation for feature analysis.

This file provides the BoundMethod class that implements a 
serializable class for binding analysis methods to feature data.
"""

from typing import Set, Callable, Any

from ..entities.feature_data import FeatureData


class BoundMethod:
    """
    Pickleable bound method without closure dependencies.
    """

    def __init__(self, feature_data: 'FeatureData', original_method: Callable, method_name: str, requires_full_data: Set[str]):
        """
        Initialize bound method.
        
        Parameters
        ----------
        feature_data : FeatureData
            Feature data object with data
        original_method : Callable
            Original calculator analysis method
        method_name : str
            Name of the method
        requires_full_data : Set[str]
            Set of method names that require full data

        Returns
        -------
        None
        """
        self.feature_data = feature_data
        self.original_method = original_method
        self.method_name = method_name
        self.requires_full_data = requires_full_data
    
    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """
        Call the bound method with automatic data selection.

        Parameters
        ----------
        args : Any
            Positional arguments for the original method
        kwargs : Any
            Keyword arguments for the original method

        Returns
        -------
        Any
            Result from the original method
        """
        # Determine which data to use
        if self.method_name in self.requires_full_data:
            data = self.feature_data.data
        else:
            # Use reduced data if available, otherwise original
            data = (
                self.feature_data.reduced_data
                if self.feature_data.reduced_data is not None
                else self.feature_data.data
            )
        
        return self.original_method(data, *args, **kwargs)
    
    def __getstate__(self):
        """
        Prepare for pickling by removing unpickleable references.

        Parameters
        ----------
        None

        Returns
        -------
        dict
            State dictionary for pickling
        """
        return {
            'method_name': self.method_name,
            'in_requires_full_data': self.method_name in self.requires_full_data
        }
    
    def __setstate__(self, state):
        """
        Restore state after unpickling.

        Parameters
        ----------
        state : dict
            State dictionary from pickling

        Returns
        -------
        None
        """
        self.method_name = state['method_name']
        self.requires_full_data = {self.method_name} if state['in_requires_full_data'] else set()
        # feature_data and original_method will be restored by parent
