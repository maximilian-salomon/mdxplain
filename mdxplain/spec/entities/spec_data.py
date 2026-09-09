# mdxplain - A Python toolkit for molecular dynamics trajectory analysis
#
# Author: Maeve Branwen Butler
# Created with assistance from GitHub Copilot (Claude Sonnet 5.0).
#
# Copyright (C) 2026 Maximilian Salomon and Maeve Branwen Butler
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
Spec data container for the one pipeline/project spec.json.

This module contains the SpecData class, the registry-driven replacement for
the old VOR-field-snapshot spec.json. See plan.md / repo memory
spec_module_design.md for the full design rationale.
"""

from typing import Any, Dict, List


class SpecData:
    """
    Pure data container for the one pipeline/project spec.json.

    Analogous to PipelineData: only ever populated/mutated through
    SpecManager, never directly. Every module is a `Dict[name, instance]`,
    where an instance is uniformly `{"modifiers": [{"type": operation_type,
    "config": {...}}, ...]}` - no separate base-config field, even the
    creating call is just the first modifier entry. `pipeline` is the one
    exception: a singleton (no name dict) since there is only ever one per
    pipeline.

    Attributes
    ----------
    pipeline : dict
        Singleton `{"modifiers": [...]}` (`pipeline_init`/`update_config`/
        `clear_all`/`add_custom_metadata` calls, in order)
    trajectory : Dict[str, dict]
        Trajectory instances keyed by name (branching allowed via modifiers
        referencing another instance as source)
    feature : Dict[str, dict]
        Feature instances keyed by name
    feature_selection : Dict[str, dict]
        Feature selector instances keyed by name
    decomposition : Dict[str, dict]
        Decomposition instances keyed by name
    clustering : Dict[str, dict]
        Clustering instances keyed by name
    data_selector : Dict[str, dict]
        Data selector instances keyed by name
    comparison : Dict[str, dict]
        Comparison instances keyed by name
    feature_importance : Dict[str, dict]
        Feature importance instances keyed by name
    structure_visualization : Dict[str, dict]
        Structure visualization instances keyed by name
    plots : Dict[str, dict]
        Plot instances keyed by name (default names, since plot calls don't
        take an explicit name parameter)
    analysis : Dict[str, dict]
        Analysis instances keyed by name (RMSD/RMSF/... services, default
        names for the same reason as `plots`)
    studies : Dict[str, Dict[str, List[str]]]
        Optional named reference bundles, grouped per module:
        `{study_name: {module_name: [instance_name, ...]}}`
    """

    MODULES = (
        "trajectory",
        "feature",
        "feature_selection",
        "decomposition",
        "clustering",
        "data_selector",
        "comparison",
        "feature_importance",
        "structure_visualization",
        "plots",
        "analysis",
    )

    def __init__(self) -> None:
        """
        Initialize an empty spec data container.

        Returns
        -------
        None
        """
        self.pipeline: Dict[str, Any] = {"modifiers": []}
        for module_name in self.MODULES:
            setattr(self, module_name, {})
        self.studies: Dict[str, Dict[str, List[str]]] = {}

