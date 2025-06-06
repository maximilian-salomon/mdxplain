"""
MDCatAFlow - MD Catalytic Analysis Flow

A high-performance Python toolkit that accelerates molecular dynamics 
trajectory analysis through automated workflows and memory-efficient processing.
"""

__version__ = "0.1.0"
__author__ = "Maximilian Salomon"

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