"""
MD Analysis Pipeline

A comprehensive toolkit for analyzing molecular dynamics trajectories.
Created with assistance from Claude-4-Sonnet and Cursor AI.
"""

__version__ = "0.1.0"
__author__ = "Maximilian Salomon"

# Import main components
from .data.TrajectoryLoader import TrajectoryLoader
from .utils.DistanceCalculator import DistanceCalculator
from .utils.ContactCalculator import ContactCalculator
from .utils.ArrayHandler import ArrayHandler

__all__ = [
    'TrajectoryLoader',
    'DistanceCalculator', 
    'ContactCalculator',
    'ArrayHandler'
] 