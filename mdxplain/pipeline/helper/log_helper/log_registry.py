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
Manual registry defining which manager/service methods are logged as pipeline
operations and which parameters are captured.

The registry data is stored in ``log_registry.json`` and resolved into the
runtime format used by the pipeline logging code. Entries are looked up by
``(owner_class, method_name)`` via ``AutoInjectProxy``/``LogHelper``.

The JSON may use arbitrary nesting for readability. A leaf entry is identified
by its ``class`` and ``module`` fields. The JSON key only needs to be unique
within its parent and has no dispatch meaning.

For callable nested services reached through ``.add.<name>``, ``method_name``
is the access name (e.g. ``"contacts"``), not ``__call__``. The class's
``__call__`` method is used internally when a real callable is needed.

Each entry may define:

- ``operation_type``: Pipeline operation type.
- ``technical_params``: Parameters captured in the log.
- ``gui_param_info``: GUI metadata.
- ``emits_tags``: Resource types written by the operation.
- ``affected_by_tags``: Resource types read by the operation.
- ``resets_tags``: Resource types fully invalidated by the operation.

Tags refer to resource types, not individual operations. The concrete resource
instance is resolved from the call parameters via ``RESOURCE_INSTANCE_PARAMS``.
Tags may contain a ``{param_name}`` placeholder for dynamic resource types.

``resets_tags`` invalidates all matching resources rather than recording a
single write. See ``LogHelper._apply_resets``.

Only operations covered by ``spec/tests/test.ipynb`` are currently registered;
the remaining entries will be added in a follow-up pass.
"""

from __future__ import annotations

import importlib
import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterator, Tuple, Type


class LogRegistry:
    """
    Namespace for the manual operation registry (build/lookup/register).

    All state lives inside the ``_build_registry`` cache (an
    ``lru_cache``-wrapped staticmethod), never as bare module-level
    variables. Use the public staticmethods below to interact with it.
    """

    _REGISTRY_JSON_PATH = Path(__file__).parent / "log_registry.json"

    @staticmethod
    def _iter_leaf_entries(node: Dict[str, Any]) -> Iterator[Dict[str, Any]]:
        """
        Recursively yield every leaf entry (dict with "module" and "class") in ``node``.

        Parameters
        ----------
        node : Dict[str, Any]
            The current node in the registry tree to inspect.

        Yields
        ------
        Dict[str, Any]
            Each leaf entry containing "module" and "class".
        """
        if "module" in node and "class" in node:
            yield node
            return
        for value in node.values():
            if isinstance(value, dict):
                yield from LogRegistry._iter_leaf_entries(value)

    @staticmethod
    def _resolve_dispatch(
        operation_type: str,
        module_path: str,
        class_name: str,
        method_name: str,
    ) -> Tuple[Type, str]:
        """
        Resolve a ``module``/``class``/``method_name`` registry entry to a
        real ``(owner_class, method_name)`` dispatch tuple.

        Parameters
        ----------
        operation_type : str
            Name of the operation being resolved, only used for error context.
        module_path : str
            Dotted module path, e.g.
            ``"mdxplain.trajectory.manager.trajectory_manager"``.
        class_name : str
            Name of the class within ``module_path``, e.g. ``"TrajectoryManager"``.
        method_name : str
            Method name as captured at runtime (may be an access-name alias
            for ``__call__`` on callable services - not necessarily a literal
            attribute of the resolved class, see module docstring).

        Returns
        -------
        Tuple[Type, str]
            The resolved ``(owner_class, method_name)`` tuple. ``method_name``
            is returned unchanged (not resolved to ``__call__``) since it must
            match exactly what is captured at runtime for dispatch lookup.

        Raises
        ------
        ValueError
            If the module cannot be imported or has no such class.
        """
        try:
            module = importlib.import_module(module_path)
        except ImportError as exc:
            raise ValueError(
                f"log_registry.json: cannot import module '{module_path}' "
                f"for dispatch of '{operation_type}'"
            ) from exc

        try:
            owner = getattr(module, class_name)
        except AttributeError as exc:
            raise ValueError(
                f"log_registry.json: module '{module_path}' has no class "
                f"'{class_name}' (dispatch of '{operation_type}')"
            ) from exc

        return owner, method_name

    @staticmethod
    @lru_cache(maxsize=1)
    def _load_raw_registry() -> Dict[str, Any]:
        """
        Load and cache the raw registry JSON file (read once, on first use).

        Returns
        -------
        Dict[str, Any]
            The raw parsed JSON, keyed by domain (plus "_resource_instance_params").
        """
        with open(LogRegistry._REGISTRY_JSON_PATH, "r", encoding="utf-8") as f:
            return json.load(f)

    @staticmethod
    @lru_cache(maxsize=1)
    def _build_registry() -> Dict[str, Dict[str, Any]]:
        """
        Build and cache the base operation registry (built once, on first use).

        Returns
        -------
        Dict[str, Dict[str, Any]]
            The single shared registry dict, keyed by operation_type. The same
            dict instance is returned on every call (cached), so
            ``LogRegistry.register_operation`` can mutate it in-place to add
            further entries.
        """
        raw_domains = LogRegistry._load_raw_registry()

        registry: Dict[str, Dict[str, Any]] = {}
        for domain_key, domain_node in raw_domains.items():
            if domain_key.startswith("_"):
                continue
            for raw_entry in LogRegistry._iter_leaf_entries(domain_node):
                method_name = raw_entry["method_name"]
                operation_type = raw_entry.get(
                    "operation_type", f"{raw_entry['class']}.{method_name}"
                )
                registry[operation_type] = {
                    "emits_tags": raw_entry.get("emits_tags", []),
                    "affected_by_tags": raw_entry.get("affected_by_tags", []),
                    "resets_tags": raw_entry.get("resets_tags", []),
                    "technical_params": raw_entry["technical_params"],
                    "dispatch": LogRegistry._resolve_dispatch(
                        operation_type,
                        raw_entry["module"],
                        raw_entry["class"],
                        method_name,
                    ),
                }
        return registry

    @staticmethod
    def get_operation_type(owner: Type, method_name: str) -> str | None:
        """
        Look up the operation_type registered for a given owner class + method.

        Parameters
        ----------
        owner : Type
            The class that owns the method (Manager or Service class).
        method_name : str
            The name of the called method.

        Returns
        -------
        str or None
            The registered operation_type, or None if this method is not logged.
        """
        dispatch = (owner, method_name)
        for operation_type, entry in LogRegistry._build_registry().items():
            if entry["dispatch"] == dispatch:
                return operation_type
        return None

    @staticmethod
    def get_instance_params(resource_type: str) -> Tuple[str, ...]:
        """
        Look up the candidate instance-name parameters of a resource type.

        Parameters
        ----------
        resource_type : str
            A resource-type tag as used in ``emits_tags``/``affected_by_tags``.

        Returns
        -------
        Tuple[str, ...]
            Ordered candidate parameter names, empty for singleton resources.
        """
        raw_domains = LogRegistry._load_raw_registry()
        return tuple(raw_domains.get("_resource_instance_params", {}).get(resource_type, ()))

    @staticmethod
    def get_registry_entry(operation_type: str) -> Dict[str, Any]:
        """
        Look up the full registry entry for a given operation_type.

        Parameters
        ----------
        operation_type : str
            Registered operation type name.

        Returns
        -------
        Dict[str, Any]
            The registry entry for ``operation_type``.
        """
        return LogRegistry._build_registry()[operation_type]

    @staticmethod
    def register_operation(operation_type: str, entry: Dict[str, Any]) -> None:
        """
        Register an operation type from a module that cannot be imported here.

        Some owner classes (e.g. ``PipelineManager``) cannot be imported by this
        module without creating a circular import, since they themselves import
        (transitively) from this module. Such modules call this function after
        their class definition instead of adding an entry directly.

        Parameters
        ----------
        operation_type : str
            Name of the operation type to register.
        entry : Dict[str, Any]
            Registry entry, see ``_build_registry`` for the expected shape.

        Returns
        -------
        None
            Updates the shared registry in-place.
        """
        LogRegistry._build_registry()[operation_type] = entry
