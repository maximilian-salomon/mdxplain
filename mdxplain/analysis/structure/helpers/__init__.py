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

"""Support helpers for structure analysis.

The helper subpackage contains streaming utilities, reference-structure
builders, and memory estimation helpers used by the structure analysis
services and calculators.
"""

from .reference_structure_helper import ReferenceStructureHelper
from .structure_calculation_helper import StructureCalculationHelper
from .residue_aggregation_helper import aggregate_residues_jit, AGGREGATOR_CODES

__all__ = [
    "ReferenceStructureHelper",
    "StructureCalculationHelper",
    "aggregate_residues_jit",
    "AGGREGATOR_CODES",
]
