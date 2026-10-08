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
Stateless helper that writes pipeline operation log entries.

``LogHelper`` holds no instance state itself - it reads/writes
``pipeline_data.log`` (a single dict grouping the operations entries
themselves plus the bookkeeping state needed to build them), analogous to
the existing ``PipelineData.add_custom_metadata``/``get_custom_metadata``
pattern.
"""

from __future__ import annotations

import inspect
from functools import wraps
from typing import Any, Callable, Dict, Tuple, Type, TYPE_CHECKING

from ...utils.operation_registry_utils import OperationRegistryUtils
from ...utils.deps_utils import DepsUtils

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
        ``log_registry.json``.

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
        if OperationRegistryUtils.get_operation_type(owner, method_name) is None:
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
    def logged(method: Callable) -> Callable:
        """
        Decorator for ``PipelineManager`` methods that never pass through
        ``AutoInjectProxy`` (it wraps the submanagers, not ``PipelineManager``
        itself), so they need to call ``LogHelper.log_call`` themselves.

        Parameters
        ----------
        method : Callable
            Unbound ``PipelineManager`` method to wrap.

        Returns
        -------
        Callable
            Wrapped method that behaves identically but additionally logs the
            call (if registered) using ``self._data`` as the pipeline data.
        """
        sig = inspect.signature(method)
        method_name = method.__name__

        @wraps(method)
        def wrapper(self, *args, **kwargs):
            result = method(self, *args, **kwargs)
            LogHelper.log_call(
                self._data, type(self), method_name, sig, (self,) + args, kwargs
            )
            return result

        return wrapper

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
        operation_type = OperationRegistryUtils.get_operation_type(owner, method_name)
        if operation_type is None:
            return

        pipeline_data.log["counters"], entry_id = DepsUtils.next_counter(
            pipeline_data.log["counters"], operation_type
        )
        global_seq = LogHelper._next_global_seq(pipeline_data)

        registry_entry = OperationRegistryUtils.get_registry_entry(operation_type)
        config = {
            param_name: bound_params[param_name]
            for param_name in registry_entry["technical_params"]
            if param_name in bound_params
        }

        depends_on, pipeline_data.log["tag_state"] = DepsUtils.resolve_and_update(
            pipeline_data.log["tag_state"], registry_entry, bound_params, entry_id,
            OperationRegistryUtils.get_instance_params,
        )

        pipeline_data.log["operations"][entry_id] = {
            "id": entry_id,
            "global_seq": global_seq,
            "type": operation_type,
            "config": config,
            "depends_on": depends_on,
        }

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
