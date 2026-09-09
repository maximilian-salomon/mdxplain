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
import re
from functools import wraps
from typing import Any, Callable, Dict, List, Tuple, Type, TYPE_CHECKING

from .log_registry import LogRegistry

if TYPE_CHECKING:
    from ...entities.pipeline_data import PipelineData


class LogHelper:
    """Stateless helper functions for writing to ``pipeline_data.log``."""

    # Resource tags implicitly read by every operation, regardless of their
    # own affected_by_tags - see _resolve_dependencies.
    _IMPLICIT_AFFECTED_BY_TAGS: Tuple[str, ...] = ("pipeline_config",)

    # Matches a single "{param_name}" placeholder inside a tag string, e.g.
    # "feature.{feature_type}" - see _resolve_tag_template.
    _TAG_PLACEHOLDER = re.compile(r"\{(\w+)\}")

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
    def logged(method: Callable) -> Callable:
        """
        Decorator for ``PipelineManager`` methods that never pass through
        ``AutoInjectProxy`` (it wraps the submanagers, not ``PipelineManager``
        itself - see ``__init__``/``log_pipeline_init`` for the same
        reasoning), so they need to call ``LogHelper.log_call`` themselves.

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
    def log_pipeline_init(
        pipeline_data: "PipelineData", owner: Type, params: Dict[str, Any]
    ) -> None:
        """
        Log the initial ``PipelineManager`` construction as an operation.

        Register pipeline_init with ``LogRegistry`` using the actual constructor
        parameters. The JSON entry is only used by ``SpecRegistryHelper``; this
        is the authoritative registration for the actual pipeline run.

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
                "emits_tags": ["pipeline_config"],
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

        depends_on = LogHelper._resolve_dependencies(
            pipeline_data, registry_entry, bound_params
        )
        LogHelper._update_tag_state(
            pipeline_data, registry_entry, bound_params, entry_id
        )
        LogHelper._apply_resets(pipeline_data, registry_entry, bound_params)

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
    def _extract_instance_key(
        resource_type: str, bound_params: Dict[str, Any]
    ) -> Any:
        """
        Determine which instance of ``resource_type`` a call refers to.

        Scans the candidate parameter names registered for the resource type
        (see ``LogRegistry.get_instance_params``) in order and returns the
        value of the first one that the call actually supplied. Used for both
        the emitting and the consuming side of dependency resolution.

        Parameters
        ----------
        resource_type : str
            Resource-type tag, as used in ``emits_tags``/``affected_by_tags``.
        bound_params : Dict[str, Any]
            Fully resolved parameters of the call.

        Returns
        -------
        Any
            None for singleton resources or calls without an instance parameter;
            the instance name for a single match; otherwise a list of instance
            names.
        """
        values = [
            bound_params[param_name]
            for param_name in LogRegistry.get_instance_params(resource_type)
            if bound_params.get(param_name) is not None
        ]
        if not values:
            return None
        if len(values) == 1:
            return values[0]
        return values

    @staticmethod
    def _instance_keys(instance_key: Any) -> list:
        """
        Normalize an extracted instance key into a list of tag-state keys.

        Parameters
        ----------
        instance_key : Any
            Result of ``_extract_instance_key``.

        Returns
        -------
        list
            Individual keys; ``[None]`` for singletons, one entry per name for
            parameters referencing several instances at once (e.g.
            ``data_selector_groups``).
        """
        if isinstance(instance_key, (list, tuple, set)):
            return list(instance_key)
        return [instance_key]

    @staticmethod
    def _resolve_tag_template(
        tag: str, bound_params: Dict[str, Any]
    ) -> Tuple[List[str], bool]:
        """
        Resolve a tag string that may contain a ``{param_name}`` placeholder.
        A tag without a placeholder resolves to itself unchanged.

        Parameters
        ----------
        tag : str
            A tag as declared in the registry, e.g. ``"clustering"`` or
            ``"feature.{feature_type}"``.
        bound_params : Dict[str, Any]
            Fully resolved parameters of the call.

        Returns
        -------
        Tuple[List[str], bool]
            ``(concrete_tags, unresolved)``. ``unresolved`` is True when a tag
            contains a placeholder whose parameter is ``None``. In this case,
            ``concrete_tags`` is empty. ``emits_tags``/``affected_by_tags`` ignore
            unresolved tags, while ``resets_tags`` resets all tags with the same
            static prefix.
        """
        match = LogHelper._TAG_PLACEHOLDER.search(tag)
        if match is None:
            return [tag], False

        param_name = match.group(1)
        value = bound_params.get(param_name)
        if value is None:
            return [], True
        if isinstance(value, (list, tuple, set)):
            return [tag.format(**{param_name: v}) for v in value], False
        return [tag.format(**{param_name: value})], False

    @staticmethod
    def _resolve_dependencies(
        pipeline_data: "PipelineData",
        registry_entry: Dict[str, Any],
        bound_params: Dict[str, Any],
    ) -> list:
        """
        Resolve structural dependencies from the current tag state.

        Find the last operation that wrote each referenced resource instance
        and resolve the implicit ``pipeline_config`` dependency.

        Parameters
        ----------
        pipeline_data : PipelineData
            Pipeline data container holding the current tag state.
        registry_entry : Dict[str, Any]
            Registry entry of the operation being logged.
        bound_params : Dict[str, Any]
            Fully resolved parameters of the call, used to identify which
            instance of each resource type is referenced.

        Returns
        -------
        list
            List of operation ids this entry directly depends on (deduped,
            order-preserving).
        """
        tag_state = pipeline_data.log["tag_state"]
        depends_on = []
        tags = list(registry_entry.get("affected_by_tags", [])) + list(
            LogHelper._IMPLICIT_AFFECTED_BY_TAGS
        )
        for tag in tags:
            concrete_tags, _unresolved = LogHelper._resolve_tag_template(
                tag, bound_params
            )
            for concrete_tag in concrete_tags:
                instances = tag_state.get(concrete_tag)
                if not instances:
                    continue
                instance_key = LogHelper._extract_instance_key(
                    concrete_tag, bound_params
                )
                for key in LogHelper._instance_keys(instance_key):
                    dependency_id = instances.get(key)
                    if dependency_id is not None and dependency_id not in depends_on:
                        depends_on.append(dependency_id)
        return depends_on

    @staticmethod
    def _update_tag_state(
        pipeline_data: "PipelineData",
        registry_entry: Dict[str, Any],
        bound_params: Dict[str, Any],
        entry_id: str,
    ) -> None:
        """
        Mark this entry as the latest writer of every resource it emits.

        Parameters
        ----------
        pipeline_data : PipelineData
            Pipeline data container holding the tag state.
        registry_entry : Dict[str, Any]
            Registry entry of the operation just logged.
        bound_params : Dict[str, Any]
            Fully resolved parameters of the call, used to identify which
            instance of each emitted resource type was written.
        entry_id : str
            The id of the entry just logged.

        Returns
        -------
        None
            Updates ``pipeline_data.log["tag_state"]`` in-place.
        """
        tag_state = pipeline_data.log["tag_state"]
        for tag in registry_entry.get("emits_tags", []):
            concrete_tags, _unresolved = LogHelper._resolve_tag_template(
                tag, bound_params
            )
            for concrete_tag in concrete_tags:
                instance_key = LogHelper._extract_instance_key(
                    concrete_tag, bound_params
                )
                for key in LogHelper._instance_keys(instance_key):
                    tag_state.setdefault(concrete_tag, {})[key] = entry_id

    @staticmethod
    def _apply_resets(
        pipeline_data: "PipelineData",
        registry_entry: Dict[str, Any],
        bound_params: Dict[str, Any],
    ) -> None:
        """
        Invalidate every instance of the resource type(s) this entry resets.

        Unlike ``emits_tags`` (which adds one instance to a resource type's
        tag_state), ``resets_tags`` clears the *entire* tag_state entry for a
        resource type - so a later operation that still references an
        instance name that existed before the reset correctly finds no
        producer, instead of the stale pre-reset one.

        Parameters
        ----------
        pipeline_data : PipelineData
            Pipeline data container holding the tag state.
        registry_entry : Dict[str, Any]
            Registry entry of the operation just logged.
        bound_params : Dict[str, Any]
            Fully resolved parameters of the call.

        Returns
        -------
        None
            Updates ``pipeline_data.log["tag_state"]`` in-place.
        """
        tag_state = pipeline_data.log["tag_state"]
        for tag in registry_entry.get("resets_tags", []):
            concrete_tags, unresolved = LogHelper._resolve_tag_template(
                tag, bound_params
            )
            if unresolved:
                prefix = tag.split("{", 1)[0]
                for existing_tag in list(tag_state):
                    if existing_tag.startswith(prefix):
                        tag_state.pop(existing_tag, None)
                continue
            for concrete_tag in concrete_tags:
                tag_state.pop(concrete_tag, None)
