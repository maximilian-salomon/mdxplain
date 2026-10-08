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

"""Replay a `SpecData` onto a real, newly built `PipelineManager`."""

from __future__ import annotations

from functools import reduce
from typing import TYPE_CHECKING, Any

from .graph_helper import GraphHelper
from .spec_builder_helper import SpecBuilderHelper
from ...utils.operation_registry_utils import OperationRegistryUtils

if TYPE_CHECKING:
    from ..entities.spec_data import SpecData

# The constructor call itself - not reachable as an attribute path on an
# already-built pipeline (see dev_scripts/populate_access_paths.py), so it
# is special-cased here instead of carrying an access_path in the registry.
_CONSTRUCTOR_OPERATION_TYPE = "pipeline_init"


class PipelineBuilderHelper:
    """Stateless helper replaying a `SpecData`'s modifiers onto a pipeline."""

    @staticmethod
    def build(spec_data: SpecData) -> Any:
        """
        Build a new, fully configured `PipelineManager` from `spec_data`.

        Linearizes every modifier across all modules into a single valid
        dependency order (``SpecBuilderHelper.resync`` + ``GraphHelper.
        layers``, the same infrastructure ``SpecManager.read_from_pipeline``
        uses in the opposite direction), then replays each one as a real
        call: ``pipeline_init`` constructs the pipeline, every other
        modifier is dispatched via its registry ``access_path``. Return
        values are collected (by reference) under each modifier's
        ``mod_name`` in ``spec_data.replay_results`` - overwritten on every
        call, since the intended usage is a single build/run per process.

        Parameters
        ----------
        spec_data : SpecData
            The spec data container to replay.

        Returns
        -------
        Any
            The newly built `PipelineManager` instance.

        Raises
        ------
        ValueError
            If no `pipeline_init` modifier is found before any other
            modifier is replayed.
        """
        SpecBuilderHelper.resync(spec_data)

        pipeline = None
        results: dict = {}
        for layer in GraphHelper.layers(spec_data.graph):
            for mod_name in sorted(layer):
                node = spec_data.graph.nodes[mod_name]
                operation_type = node["type"]
                config = node["config"]

                if operation_type == _CONSTRUCTOR_OPERATION_TYPE:
                    from ...pipeline.manager.pipeline_manager import (
                        PipelineManager,
                    )

                    pipeline = PipelineManager(**config)
                    results[mod_name] = pipeline
                    continue

                if pipeline is None:
                    raise ValueError(
                        f"Modifier '{mod_name}' ({operation_type}) replayed "
                        f"before '{_CONSTRUCTOR_OPERATION_TYPE}' - no "
                        f"pipeline to call it on."
                    )

                registry_entry = OperationRegistryUtils.get_registry_entry(
                    operation_type
                )
                access_path = registry_entry["access_path"]
                target = reduce(getattr, access_path.split("."), pipeline)
                call_kwargs = PipelineBuilderHelper._resolve_call_kwargs(
                    registry_entry["technical_params"], config
                )
                results[mod_name] = target(**call_kwargs)

        spec_data.replay_results = results
        return pipeline

    @staticmethod
    def _resolve_call_kwargs(
        technical_params: dict, config: dict
    ) -> dict:
        """
        Expand a ``**kwargs``-style technical_param into a real call's kwargs.

        A ``technical_params`` entry marked ``"kind": "VAR_KEYWORD"`` is not a
        real parameter of the target method, but the name of its ``**kwargs``
        catch-all (see ``dev_scripts/check_log_registry.py``) - passing it
        through as a literal ``name=value`` keyword would nest ``value`` one
        level too deep inside the catch-all itself. Its dict value is spread
        into the call instead.

        Parameters
        ----------
        technical_params : dict
            The operation's ``technical_params`` registry entry.
        config : dict
            The modifier's stored config values.

        Returns
        -------
        dict
            Kwargs ready to call the target method with (``target(**result)``).
        """
        var_keyword_name = next(
            (
                name
                for name, info in technical_params.items()
                if info.get("kind") == "VAR_KEYWORD"
            ),
            None,
        )
        if var_keyword_name is None or var_keyword_name not in config:
            return config

        call_kwargs = {k: v for k, v in config.items() if k != var_keyword_name}
        call_kwargs.update(config[var_keyword_name])
        return call_kwargs
