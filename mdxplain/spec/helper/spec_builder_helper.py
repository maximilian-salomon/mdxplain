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
Core add()-logic: instance name resolution and modifier-list mutation.

Used by ``SpecManager.add()`` for both the manual/GUI construction path and
(reused internally) the Graph->JSON translation - see repo memory
spec_module_design.md for the full design.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple


class SpecBuilderHelper:
    """
    Static helper resolving instance names and mutating `Dict[name, instance]` modules.

    Examples
    --------
    >>> module = {}
    >>> name = SpecBuilderHelper.resolve_name(module, "dbscan", None, ("cluster_name", "name"), {"cluster_name": "c1"})
    >>> SpecBuilderHelper.add_modifier(module, name, "DBSCANAddService.dbscan", {"cluster_name": "c1", "eps": 0.5})
    >>> module
    {'c1': {'modifiers': [{'type': 'DBSCANAddService.dbscan', 'config': {'cluster_name': 'c1', 'eps': 0.5}}]}}
    """

    @staticmethod
    def resolve_name(
        target: Dict[str, Any],
        method_name: str,
        name: Optional[str],
        instance_params: Tuple[str, ...],
        kwargs: Dict[str, Any],
    ) -> str:
        """
        Resolve the instance name a call should be stored under in ``target``.

        Priority: explicit ``name`` > first ``instance_params`` candidate
        present in ``kwargs`` > default derived from ``method_name`` (a
        numeric suffix is only added if that default name is already taken,
        e.g. ``slice_traj``, ``slice_traj_1``, ``slice_traj_2``, ...).

        Parameters
        ----------
        target : Dict[str, Any]
            The module's `Dict[name, instance]`, used to de-duplicate the
            derived default name against existing entries.
        method_name : str
            Method name (the part of `operation_type` after the last ".").
        name : str, optional
            Explicit name override, if the caller supplied one.
        instance_params : Tuple[str, ...]
            Ordered candidate parameter names for this resource type (see
            ``SpecRegistryHelper.get_instance_params``).
        kwargs : Dict[str, Any]
            The call's config values.

        Returns
        -------
        str
            The resolved instance name.
        """
        if name is not None:
            return name

        for candidate in instance_params:
            if kwargs.get(candidate) is not None:
                return kwargs[candidate]

        if method_name not in target:
            return method_name
        n = 1
        while f"{method_name}_{n}" in target:
            n += 1
        return f"{method_name}_{n}"

    @staticmethod
    def add_modifier(
        target: Dict[str, Any],
        resolved_name: str,
        operation_type: str,
        config: Dict[str, Any],
    ) -> None:
        """
        Append a modifier entry to a named instance, creating it if new.

        Parameters
        ----------
        target : Dict[str, Any]
            The module's `Dict[name, instance]`, mutated in place.
        resolved_name : str
            The instance name to add/append to (see ``resolve_name``).
        operation_type : str
            Registered operation type name (``"ClassName.method_name"``).
        config : Dict[str, Any]
            The call's config values, stored as-is.

        Returns
        -------
        None
            Mutates ``target`` in place.
        """
        if resolved_name not in target:
            target[resolved_name] = {"modifiers": []}
        target[resolved_name]["modifiers"].append({"type": operation_type, "config": config})

    @staticmethod
    def add_singleton_modifier(
        singleton: Dict[str, Any], operation_type: str, config: Dict[str, Any]
    ) -> None:
        """
        Append a modifier entry to a singleton instance (e.g. ``pipeline``).

        Parameters
        ----------
        singleton : Dict[str, Any]
            A single instance dict (``{"modifiers": [...]}``), not a
            `Dict[name, instance]` module.
        operation_type : str
            Registered operation type name (``"ClassName.method_name"``).
        config : Dict[str, Any]
            The call's config values, stored as-is.

        Returns
        -------
        None
            Mutates ``singleton`` in place.
        """
        singleton["modifiers"].append({"type": operation_type, "config": config})
