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
Lightweight, JSON-only reader for mdxplain's log_registry.json.

Deliberately separate from ``mdxplain.pipeline.helper.log_helper.log_registry.LogRegistry``:
that class resolves real dispatch classes on first use (importing the full
mdxplain domain stack, including heavy deps like mdtraj/sklearn), which
``spec/`` must avoid to stay usable by a lightweight GUI without installing
mdxplain's runtime dependencies (see repo memory spec_module_design.md).
This reader only ever parses the raw JSON file - no ``importlib``, no
``mdxplain.*`` import.

The ``operation_type`` keys used here (``"ClassName.method_name"``) match
exactly what ``LogHelper.log_operation`` writes as the ``"type"`` field of a
real pipeline_data.log entry, so entries logged by a real pipeline run and
entries built manually via ``SpecManager.add()`` resolve to the same
registry data.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterator, Tuple


class SpecRegistryHelper:
    """
    Static helper reading ``log_registry.json`` directly, without dispatch resolution.

    Examples
    --------
    >>> SpecRegistryHelper.get_domain("DBSCANAddService.dbscan")
    'clustering'
    >>> SpecRegistryHelper.get_instance_params("clustering")
    ('cluster_name', 'clustering_name', 'name')
    """

    _REGISTRY_JSON_PATH = (
        Path(__file__).resolve().parents[2]
        / "pipeline"
        / "helper"
        / "log_helper"
        / "log_registry.json"
    )

    @staticmethod
    @lru_cache(maxsize=1)
    def _load_raw_registry() -> Dict[str, Any]:
        """
        Load and cache the raw registry JSON file (read once, on first use).

        Returns
        -------
        Dict[str, Any]
            The raw parsed JSON, keyed by domain (plus ``_resource_instance_params``).
        """
        with open(SpecRegistryHelper._REGISTRY_JSON_PATH, "r", encoding="utf-8") as f:
            return json.load(f)

    @staticmethod
    def _iter_leaf_entries(
        node: Dict[str, Any],
    ) -> Iterator[Dict[str, Any]]:
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
                yield from SpecRegistryHelper._iter_leaf_entries(value)

    @staticmethod
    @lru_cache(maxsize=1)
    def _build_registry() -> Dict[str, Dict[str, Any]]:
        """
        Build and cache the flat operation_type -> entry registry (built once).

        Returns
        -------
        Dict[str, Dict[str, Any]]
            Keyed by ``operation_type`` (``"ClassName.method_name"``, or an
            explicit ``"operation_type"`` override on the leaf entry - e.g.
            ``"pipeline_init"``, whose real logged type is not derivable
            from class+method alone). Each entry carries ``domain``,
            ``emits_tags``, ``affected_by_tags``, ``resets_tags`` and
            ``technical_params`` - no ``dispatch`` tuple, unlike mdxplain's
            own ``LogRegistry``.
        """
        raw_domains = SpecRegistryHelper._load_raw_registry()

        registry: Dict[str, Dict[str, Any]] = {}
        for domain_key, domain_node in raw_domains.items():
            if domain_key.startswith("_"):
                continue
            for raw_entry in SpecRegistryHelper._iter_leaf_entries(domain_node):
                operation_type = raw_entry.get(
                    "operation_type", f"{raw_entry['class']}.{raw_entry['method_name']}"
                )
                registry[operation_type] = {
                    "domain": domain_key,
                    "emits_tags": raw_entry.get("emits_tags", []),
                    "affected_by_tags": raw_entry.get("affected_by_tags", []),
                    "resets_tags": raw_entry.get("resets_tags", []),
                    "technical_params": raw_entry["technical_params"],
                }
        return registry

    @staticmethod
    def get_entry(operation_type: str) -> Dict[str, Any]:
        """
        Look up the full registry entry for a given operation_type.

        Parameters
        ----------
        operation_type : str
            Registered operation type name (``"ClassName.method_name"``).

        Returns
        -------
        Dict[str, Any]
            The registry entry for ``operation_type``.

        Raises
        ------
        KeyError
            If ``operation_type`` is not registered.
        """
        return SpecRegistryHelper._build_registry()[operation_type]

    @staticmethod
    def get_domain(operation_type: str) -> str:
        """
        Look up the top-level domain of a registered operation_type.

        Parameters
        ----------
        operation_type : str
            Registered operation type name (``"ClassName.method_name"``).

        Returns
        -------
        str
            The domain (top-level key in ``log_registry.json``) this
            operation belongs to, e.g. ``"clustering"``.

        Raises
        ------
        KeyError
            If ``operation_type`` is not registered.
        """
        return SpecRegistryHelper.get_entry(operation_type)["domain"]

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
        raw_domains = SpecRegistryHelper._load_raw_registry()
        return tuple(raw_domains.get("_resource_instance_params", {}).get(resource_type, ()))
