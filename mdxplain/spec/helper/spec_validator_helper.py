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
Required-param/cross-reference validation and structural spec comparison.

Covers the two independent validation concerns used while building a spec
(``check_required_params``/``check_cross_references``, invoked from
``SpecManager.validate``) as well as ``compare_specs``, an order-independent
structural diff between two specs (see repo memory spec_module_design.md).
"""

from __future__ import annotations

import json
from collections import Counter
from typing import Any, Dict, List, Sequence, Union, TYPE_CHECKING

from ...utils.operation_registry_utils import OperationRegistryUtils
from .spec_io_helper import SpecIOHelper

if TYPE_CHECKING:
    from ..entities.spec_data import SpecData


class SpecValidatorHelper:
    """Static helper validating required params and cross-module references."""

    # also check if depend on tags exist in modifiers

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
                f"Missing required parameter(s) {missing} for "
                f"'{operation_type}'"
            )

    @staticmethod
    def check_cross_references(
        spec_data: Any,
        entry: Dict[str, Any],
        own_resource_type: str,
        kwargs: Dict[str, Any],
    ) -> None:
        """
        Raise if ``kwargs`` references a not-(yet)-existing instance in
        another module.

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
            if resource_type != own_resource_type
            and hasattr(spec_data, resource_type)
        ]
        if not affected_types:
            return

        candidate_to_types: Dict[str, list] = {}
        for resource_type in affected_types:
            for candidate in OperationRegistryUtils.get_instance_params(
                resource_type
            ):
                if candidate == "name":
                    continue
                candidate_to_types.setdefault(candidate, []).append(
                    resource_type
                )

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
                        f"'{candidate}={referenced_name!r}' does not "
                        f"reference an existing instance in any of "
                        f"{resource_types}"
                    )

    @staticmethod
    def _modifier_key(
        modifier: Dict[str, Any], ignore_keys: Sequence[str]
    ) -> str:
        """Canonical, order-independent identity of a modifier's
        logical content."""
        content = {k: v for k, v in modifier.items() if k not in ignore_keys}
        return json.dumps(content, sort_keys=True, default=str)

    @staticmethod
    def _diff_modifiers(
        expected_modifiers: List[Dict[str, Any]],
        actual_modifiers: List[Dict[str, Any]],
        ignore_keys: Sequence[str],
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Multiset-diff two modifier lists by content, independent of order."""
        by_key_expected = {
            SpecValidatorHelper._modifier_key(m, ignore_keys): m
            for m in expected_modifiers
        }
        by_key_actual = {
            SpecValidatorHelper._modifier_key(m, ignore_keys): m
            for m in actual_modifiers
        }
        expected_counts = Counter(
            SpecValidatorHelper._modifier_key(m, ignore_keys)
            for m in expected_modifiers
        )
        actual_counts = Counter(
            SpecValidatorHelper._modifier_key(m, ignore_keys)
            for m in actual_modifiers
        )
        missing = expected_counts - actual_counts
        extra = actual_counts - expected_counts
        return {
            "missing_modifiers": [
                by_key_expected[key] for key in missing.elements()
            ],
            "extra_modifiers": [by_key_actual[key] for key in extra.elements()],
        }

    @staticmethod
    def compare_specs(
        expected: Union["SpecData", Dict[str, Any]],
        actual: Union["SpecData", Dict[str, Any]],
        ignore_keys: Sequence[str] = ("mod_name", "depends_on", "order"),
        by_instance: bool = False,
    ) -> Dict[str, Any]:
        """
        Structurally compare two specs, independent of modifier order.

        Unlike comparing ``get_all_modifiers()`` lists (order- and
        position-sensitive), this matches modifiers by their logical content
        (``type``/``config``, everything except ``ignore_keys``), since specs
        built via different paths (e.g. graph-layer traversal in
        ``SpecManager.read_from_pipeline`` vs. sequential ``add()`` calls) can
        reconstruct the same operations in a different order without being
        wrong - see ``LogGraphHelper.layers``.

        Parameters
        ----------
        expected : SpecData or Dict[str, Any]
            Reference spec, either a ``SpecData`` instance or a plain dict (as
            produced by ``SpecIOHelper.to_dict``/``json.load``) - one key per
            domain, each holding that domain's
            ``Dict[instance_name, instance]``.
        actual : SpecData or Dict[str, Any]
            Spec to compare against ``expected``, same accepted formats.
        ignore_keys : Sequence[str], default=("mod_name", "depends_on", "order")
            Modifier keys excluded from the content comparison because they are
            build-path-specific (e.g. counters assigned in call order) rather
            than part of the logical operation.
        by_instance : bool, default=False
            When False (default), modifiers are compared per domain only,
            ignoring which instance they are attached to - auto-derived
            instance names (e.g. ``landscape`` vs. ``landscape_1``, assigned
            by ``SpecBuilderHelper.resolve_instance_name`` based on
            processing order) are just as build-path-specific as ``mod_name``,
            so requiring an exact instance-name match would produce false
            mismatches for equivalent specs. Set to True to additionally
            require modifiers to be attached to identically-named instances.

        Returns
        -------
        Dict[str, Any]
            ``{"equal": bool, "domains": {domain_name: diff}}``. Only domains
            that actually differ are included. With ``by_instance=False``,
            each domain's ``diff`` is ``{"missing_modifiers": [...],
            "extra_modifiers": [...]}``. With ``by_instance=True``, it also has
            ``"missing_instances"``/``"extra_instances"`` (instance names) and
            ``"instances"``: ``{instance_name: {"missing_modifiers": [...],
            "extra_modifiers": [...]}}`` for instances present in both but with
            differing modifiers. ``"studies"`` (if present) is compared as
            plain values instead, since it holds name-reference lists rather
            than modifiers.
        """
        if not isinstance(expected, dict):
            expected = SpecIOHelper.to_dict(expected)
        if not isinstance(actual, dict):
            actual = SpecIOHelper.to_dict(actual)

        domain_names = sorted(set(expected) | set(actual))
        domains_diff: Dict[str, Any] = {}

        for domain_name in domain_names:
            expected_domain = expected.get(domain_name, {})
            actual_domain = actual.get(domain_name, {})

            if domain_name == "studies":
                if expected_domain != actual_domain:
                    domains_diff[domain_name] = {
                        "expected": expected_domain,
                        "actual": actual_domain,
                    }
                continue

            # The "pipeline" domain is a singleton instance, not a
            # Dict[name, instance] - normalize it to the same shape.
            if domain_name == "pipeline":
                expected_instances = {"pipeline": expected_domain}
                actual_instances = {"pipeline": actual_domain}
            else:
                expected_instances = expected_domain
                actual_instances = actual_domain

            if not by_instance:
                expected_modifiers = [
                    m
                    for inst in expected_instances.values()
                    for m in inst.get("modifiers", [])
                ]
                actual_modifiers = [
                    m
                    for inst in actual_instances.values()
                    for m in inst.get("modifiers", [])
                ]
                diff = SpecValidatorHelper._diff_modifiers(
                    expected_modifiers, actual_modifiers, ignore_keys
                )
                if diff["missing_modifiers"] or diff["extra_modifiers"]:
                    domains_diff[domain_name] = diff
                continue

            missing_instances = sorted(
                set(expected_instances) - set(actual_instances)
            )
            extra_instances = sorted(
                set(actual_instances) - set(expected_instances)
            )

            instances_diff: Dict[str, Any] = {}
            for instance_name in sorted(
                set(expected_instances) & set(actual_instances)
            ):
                diff = SpecValidatorHelper._diff_modifiers(
                    expected_instances[instance_name].get("modifiers", []),
                    actual_instances[instance_name].get("modifiers", []),
                    ignore_keys,
                )
                if diff["missing_modifiers"] or diff["extra_modifiers"]:
                    instances_diff[instance_name] = diff

            if missing_instances or extra_instances or instances_diff:
                domains_diff[domain_name] = {
                    "missing_instances": missing_instances,
                    "extra_instances": extra_instances,
                    "instances": instances_diff,
                }

        return {"equal": not domains_diff, "domains": domains_diff}
