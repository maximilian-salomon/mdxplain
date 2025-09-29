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

"""Public structure analysis services.

The :mod:`mdxplain.analysis.structure.services` package exposes the lazily
loaded structure analysis facade together with the concrete RMSD and RMSF
services. All services expect a pipeline context injected via AutoInject.
"""

from .structure_analysis_service import StructureAnalysisService
from .rmsd_facade import RMSDFacade
from .rmsd_variant_service import RMSDVariantService
from .rmsf_facade import RMSFFacade
from .rmsf_variant_facade import RMSFVariantFacade
from .rmsf_per_atom_service import RMSFPerAtomService
from .rmsf_per_residue_service import RMSFPerResidueService

__all__ = [
    "StructureAnalysisService",
    "RMSDFacade",
    "RMSDVariantService",
    "RMSFFacade",
    "RMSFVariantFacade",
    "RMSFPerAtomService",
    "RMSFPerResidueService",
]
