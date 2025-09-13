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

"""Simple test helper that bypasses complex trajectory setup."""

import numpy as np
from mdxplain.pipeline.managers.pipeline_manager import PipelineManager
from mdxplain.feature.feature_type.distances import Distances
from mdxplain.feature.feature_type.contacts import Contacts


class SimpleTestHelper:
    """Helper for creating simple test pipelines with mock data."""
    
    @staticmethod
    def create_pipeline_with_mock_distances(n_frames=100, n_pairs=10, seed=42):
        """Create pipeline with mock distance data."""
        np.random.seed(seed)
        
        pipeline = PipelineManager()
        
        # Create mock distance data directly
        distance_data = np.random.rand(n_frames, n_pairs) * 10 + 2.0  # 2-12 Angstrom
        
        # Inject mock data directly into pipeline
        distances = Distances()
        distances.init_calculator()
        
        # Store in feature data structure
        feature_data = pipeline._data.feature_data.setdefault('distances', {})
        feature_data['data'] = distance_data
        feature_data['feature_metadata'] = {'n_pairs': n_pairs, 'pairs': list(range(n_pairs))}
        
        # Create mock feature object
        pipeline.feature.distances = MockFeatureData(distance_data)
        
        return pipeline
    
    @staticmethod 
    def create_pipeline_with_mock_contacts(n_frames=100, n_pairs=10, seed=42):
        """Create pipeline with mock contact data."""
        np.random.seed(seed)
        
        pipeline = PipelineManager()
        
        # Create mock binary contact data
        contact_data = np.random.randint(0, 2, size=(n_frames, n_pairs))
        
        # Store in pipeline
        feature_data = pipeline._data.feature_data.setdefault('contacts', {})
        feature_data['data'] = contact_data
        feature_data['feature_metadata'] = {'n_pairs': n_pairs, 'pairs': list(range(n_pairs))}
        
        pipeline.feature.contacts = MockFeatureData(contact_data)
        
        return pipeline


class MockFeatureData:
    """Mock feature data object for testing analysis methods."""
    
    def __init__(self, data):
        self.data = data
        self.reduced_data = None
        self.analysis = MockAnalysis(data)
        

class MockAnalysis:
    """Mock analysis object with basic methods."""
    
    def __init__(self, data):
        self._data = data
        
    def mean(self):
        return np.mean(self._data)
        
    def std(self):
        return np.std(self._data)
        
    def min(self):
        return np.min(self._data)
        
    def max(self):
        return np.max(self._data)
        
    def sum(self):
        return np.sum(self._data)