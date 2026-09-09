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
Required-param and cross-reference validation for spec.json construction.

Used by ``SpecManager.add()`` incrementally, and (whole-document) after
``SpecManager.read_json()`` - see repo memory spec_module_design.md.
"""

from __future__ import annotations

from typing import Any, Dict

from .spec_registry_helper import SpecRegistryHelper


class SpecValidatorHelper:
    """Static helper validating required params and cross-module references."""

    @staticmethod
    def check_required_params(
        operation_type: str, entry: Dict[str, Any], kwargs: Dict[str, Any]
    ) -> None:
        """
        Raise if a required technical_param is missing from ``kwargs``.

        Always decides via the entry's explicit ``"required"`` flag - never
        via ``default is None`` (a default value of ``None`` is a legitimate,
        optional default, not a signal that the param is required).

        Parameters
        ----------
        operation_type : str
            Registered operation type name, used for the error message.
        entry : Dict[str, Any]
            Registry entry (see ``SpecRegistryHelper.get_entry``).
        kwargs : Dict[str, Any]
            The call's config values.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If one or more required params are missing from ``kwargs``.
        """
        missing = [
            param_name
            for param_name, param_info in entry["technical_params"].items()
            if param_info.get("required", False) and param_name not in kwargs
        ]
        if missing:
            raise ValueError(
                f"Missing required parameter(s) {missing} for '{operation_type}'"
            )

    @staticmethod
    def check_cross_references(
        spec_data: Any, entry: Dict[str, Any], own_resource_type: str, kwargs: Dict[str, Any]
    ) -> None:
        """
        Raise if ``kwargs`` references a not-(yet)-existing instance in another module.

        Only checks resource types this specific operation actually declares
        as a read-dependency (``entry["affected_by_tags"]``). Some
        instance-name parameters are overloaded across resource types (e.g.
        ``selection_name`` can name either a ``feature_selection`` or a
        ``decomposition``) - a value is accepted if it exists in *any* of the
        resource types that declare it as a candidate, since which one it
        actually is cannot be told from the parameter name alone. The
        generic ``"name"`` candidate is never checked as a cross-reference -
        it is always reserved for the call's own instance identity (used by
        ``SpecBuilderHelper.resolve_name``), not a pointer into another
        module, even though it also appears as a naming fallback in other
        resource types' candidate lists.

        Parameters
        ----------
        spec_data : SpecData
            The spec data container to check references against.
        entry : Dict[str, Any]
            Registry entry (see ``SpecRegistryHelper.get_entry``), whose
            ``affected_by_tags`` list the resource types to check.
        own_resource_type : str
            The resource type this call itself emits (excluded from the
            check - a call's own identity param is not a reference).
        kwargs : Dict[str, Any]
            The call's config values.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If a referenced instance name does not exist in any of the
            resource types that declare its parameter name as a candidate.
        """
        affected_types = [
            resource_type
            for resource_type in entry.get("affected_by_tags", [])
            if resource_type != own_resource_type and hasattr(spec_data, resource_type)
        ]
        if not affected_types:
            return

        candidate_to_types: Dict[str, list] = {}
        for resource_type in affected_types:
            for candidate in SpecRegistryHelper.get_instance_params(resource_type):
                if candidate == "name":
                    continue
                candidate_to_types.setdefault(candidate, []).append(resource_type)

        for candidate, resource_types in candidate_to_types.items():
            value = kwargs.get(candidate)
            if value is None:
                continue
            referenced_names = value if isinstance(value, list) else [value]
            for referenced_name in referenced_names:
                if not any(
                    referenced_name in getattr(spec_data, resource_type)
                    for resource_type in resource_types
                ):
                    raise ValueError(
                        f"'{candidate}={referenced_name!r}' does not reference an "
                        f"existing instance in any of {resource_types}"
                    )
