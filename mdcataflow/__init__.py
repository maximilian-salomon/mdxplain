"""
MD Analysis Pipeline

A comprehensive toolkit for analyzing molecular dynamics trajectories.
"""

__version__ = "1.0.0"
__author__ = "MD Analysis Team"

# Import main components
from .data.TrajectoryLoader import TrajectoryLoader
from .utils.DistanceCalculator import DistanceCalculator
from .utils.ContactCalculator import ContactCalculator
from .utils.ArrayConverter import ArrayHandler

__all__ = [
    'TrajectoryLoader',
    'DistanceCalculator', 
    'ContactCalculator',
    'ArrayHandler'
] 