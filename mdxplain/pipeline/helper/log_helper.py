# mdxplain - A Python toolkit for molecular dynamics trajectory analysis
#
# Author: Maximilian Salomon
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
Stateless helper that writes pipeline operation log entries.

``LogHelper`` holds no instance state itself - it reads/writes
``pipeline_data.log`` (a single dict grouping the operations entries
themselves plus the bookkeeping state needed to build them), analogous to
the existing ``PipelineData.add_custom_metadata``/``get_custom_metadata``
pattern.

Called from ``AutoInjectProxy`` (see
``mdxplain/pipeline/manager/auto_inject_proxy.py``) and
``LoggingServiceProxy`` (see
``mdxplain/pipeline/manager/logging_service_proxy.py``) after a logged
manager/service method has been executed successfully.
"""

from __future__ import annotations

import inspect
from typing import Any, Dict, Tuple, Type, TYPE_CHECKING

from .log_registry import LogRegistry

if TYPE_CHECKING:
    from ..entities.pipeline_data import PipelineData


class LogHelper:
    """Stateless helper functions for writing to ``pipeline_data.log``."""

    @staticmethod
    def log_call(
        pipeline_data: "PipelineData",
        owner: Type,
        method_name: str,
        sig: inspect.Signature,
        args: Tuple[Any, ...],
        kwargs: Dict[str, Any],
    ) -> None:
        """
        Resolve full call parameters (defaults applied) and log via LogHelper.

        Does nothing if (owner, method_name) is not registered in
        ``log_registry``. Shared entry point for both ``AutoInjectProxy``
        and ``LoggingServiceProxy``.

        Parameters
        ----------
        pipeline_data : PipelineData
            Pipeline data container to log into.
        owner : type
            Manager or Service class that owns the called method.
        method_name : str
            Name of the called method ("__call__" for callable services).
        sig : inspect.Signature
            Signature of the (bound) method, used to resolve defaults.
        args : tuple
            Positional arguments actually passed to the method.
        kwargs : dict
            Keyword arguments actually passed to the method.

        Returns
        -------
        None
            Writes the log entry into ``pipeline_data.log`` in-place (if
            registered).
        """
        if LogRegistry.get_operation_type(owner, method_name) is None:
            return
        try:
            bound = sig.bind(*args, **kwargs)
            bound.apply_defaults()
            bound_params = dict(bound.arguments)
        except TypeError:
            bound_params = dict(kwargs)
        bound_params.pop("pipeline_data", None)
        bound_params.pop("self", None)
        LogHelper.log_operation(pipeline_data, owner, method_name, bound_params)

    @staticmethod
    def log_pipeline_init(
        pipeline_data: "PipelineData", owner: Type, params: Dict[str, Any]
    ) -> None:
        """
        Log the initial ``PipelineManager`` construction as an operation.

        ``PipelineManager.__init__`` cannot go through ``AutoInjectProxy``
        (it constructs ``pipeline_data`` itself), so it calls this directly
        instead. Registers the "pipeline_init" operation type on first use
        via ``LogRegistry.register_operation`` (idempotent - simply
        overwrites the same entry on subsequent calls), then logs the call
        like any other operation.

        Parameters
        ----------
        pipeline_data : PipelineData
            Pipeline data container to log into.
        owner : Type
            The ``PipelineManager`` class.
        params : Dict[str, Any]
            Fully resolved constructor parameters (``self`` excluded).

        Returns
        -------
        None
            Writes the log entry into ``pipeline_data.log`` in-place.
        """
        LogRegistry.register_operation(
            "pipeline_init",
            {
                "dispatch": (owner, "__init__"),
                "emits_tags": [],
                "affected_by_tags": [],
                "technical_params": list(params.keys()),
                "gui_param_info": {},
            },
        )
        LogHelper.log_operation(pipeline_data, owner, "__init__", params)

    @staticmethod
    def log_operation(
        pipeline_data: "PipelineData",
        owner: Type,
        method_name: str,
        bound_params: Dict[str, Any],
    ) -> None:
        """
        Log a single pipeline operation, if it is registered.

        Parameters
        ----------
        pipeline_data : PipelineData
            Pipeline data container holding the operations log.
        owner : Type
            The Manager or Service class that owns the called method.
        method_name : str
            Name of the called method (use "__call__" for callable services).
        bound_params : Dict[str, Any]
            Fully resolved parameters of the call (defaults already applied,
            ``self``/``pipeline_data`` excluded).

        Returns
        -------
        None
            Writes the entry into ``pipeline_data.log["operations"]``
            in-place. Silently does nothing if (owner, method_name) is not
            registered.
        """
        operation_type = LogRegistry.get_operation_type(owner, method_name)
        if operation_type is None:
            return

        entry_id = LogHelper._next_id(pipeline_data, operation_type)
        global_seq = LogHelper._next_global_seq(pipeline_data)

        registry_entry = LogRegistry.get_registry_entry(operation_type)
        config = {
            param_name: bound_params[param_name]
            for param_name in registry_entry["technical_params"]
            if param_name in bound_params
        }

        depends_on = LogHelper._resolve_dependencies(pipeline_data, registry_entry)
        LogHelper._update_tag_state(pipeline_data, registry_entry, entry_id)

        pipeline_data.log["operations"][entry_id] = {
            "id": entry_id,
            "global_seq": global_seq,
            "type": operation_type,
            "config": config,
            "depends_on": depends_on,
        }

    @staticmethod
    def _next_id(pipeline_data: "PipelineData", operation_type: str) -> str:
        """
        Compute and reserve the next global per-type id (e.g. ``dbscan_2``).

        Parameters
        ----------
        pipeline_data : PipelineData
            Pipeline data container holding the type counters.
        operation_type : str
            Registered operation type name.

        Returns
        -------
        str
            The reserved id, e.g. ``"dbscan_2"``.
        """
        counters = pipeline_data.log["counters"]
        next_n = counters.get(operation_type, 0) + 1
        counters[operation_type] = next_n
        return f"{operation_type}_{next_n}"

    @staticmethod
    def _next_global_seq(pipeline_data: "PipelineData") -> int:
        """
        Compute and reserve the next monotonic global sequence number.

        Parameters
        ----------
        pipeline_data : PipelineData
            Pipeline data container holding the global sequence counter.

        Returns
        -------
        int
            The reserved global sequence number.
        """
        pipeline_data.log["global_seq"] += 1
        return pipeline_data.log["global_seq"]

    @staticmethod
    def _resolve_dependencies(
        pipeline_data: "PipelineData", registry_entry: Dict[str, Any]
    ) -> list:
        """
        Resolve direct structural dependencies via the current tag state.

        Parameters
        ----------
        pipeline_data : PipelineData
            Pipeline data container holding the current tag -> id state.
        registry_entry : Dict[str, Any]
            Registry entry of the operation being logged.

        Returns
        -------
        list
            List of operation ids this entry directly depends on (deduped,
            order-preserving).
        """
        tag_state = pipeline_data.log["tag_state"]
        depends_on = []
        for tag in registry_entry["affected_by_tags"]:
            dependency_id = tag_state.get(tag)
            if dependency_id is not None and dependency_id not in depends_on:
                depends_on.append(dependency_id)
        return depends_on

    @staticmethod
    def _update_tag_state(
        pipeline_data: "PipelineData", registry_entry: Dict[str, Any], entry_id: str
    ) -> None:
        """
        Update the tag -> latest-id state after logging an entry.

        Parameters
        ----------
        pipeline_data : PipelineData
            Pipeline data container holding the tag state.
        registry_entry : Dict[str, Any]
            Registry entry of the operation just logged.
        entry_id : str
            The id of the entry just logged.

        Returns
        -------
        None
            Updates ``pipeline_data.log["tag_state"]`` in-place.
        """
        for tag in registry_entry["emits_tags"]:
            pipeline_data.log["tag_state"][tag] = entry_id
