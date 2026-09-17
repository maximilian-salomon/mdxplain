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
Utility functions for dependency resolution and instance key management.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Tuple, Callable

class DepsUtils:
    """
    Stateless utility class for managing dependency resolution and instance keys.

    This class provides methods to compute operation counters, extract instance
    keys from method parameters, and normalize them into lists suitable for
    dependency tracking.
    """

    # Resource tags implicitly read by every operation, regardless of their
    # own affected_by_tags - see _resolve_dependencies.
    _IMPLICIT_AFFECTED_BY_TAGS: Tuple[str, ...] = ("pipeline_config",)

    # Matches a single "{param_name}" placeholder inside a tag string, e.g.
    # "feature.{feature_type}" - see _resolve_tag_template.
    _TAG_PLACEHOLDER = re.compile(r"\{(\w+)\}")

    @staticmethod
    def next_counter(counters: Dict[str, int], operation_type: str) -> Tuple[Dict[str, int], str]:
        """
        Compute and reserve the next counter for the given operation type (e.g. ``dbscan_2``).

        Parameters
        ----------
        counters : Dict[str, int]
            Dictionary holding the type counters.
        operation_type : str
            Registered operation type name.

        Returns
        -------
        Tuple[Dict[str, int], str]
            The updated counters dictionary and the reserved id, e.g. ``"dbscan_2"``.
        """
        next_n = counters.get(operation_type, 0) + 1
        counters[operation_type] = next_n
        return counters, f"{operation_type}_{next_n}"

    @staticmethod
    def _extract_instance_key(
        resource_type: str,
        bound_params: Dict[str, Any],
        get_instance_params: Callable[[str], List[str]],
    ) -> Any:
        """
        Determine which instance of ``resource_type`` a call refers to.

        Scans the candidate parameter names registered for the resource type
        (see ``LogRegistry.get_instance_params``) in order and returns the
        value of the first one that the call actually supplied. Used for both
        the emitting and the consuming side of dependency resolution.

        Parameters
        ----------
        bound_params : Dict[str, Any]
            Fully resolved parameters of the call.
        get_instance_params : Callable[[str], List[str]]
            Function to retrieve candidate parameter names for the resource instance.

        Returns
        -------
        Any
            None for singleton resources or calls without an instance parameter;
            the instance name for a single match; otherwise a list of instance
            names.
        """
        values = [
            bound_params[param_name]
            for param_name in get_instance_params(resource_type)
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
        match = DepsUtils._TAG_PLACEHOLDER.search(tag)
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
    def resolve_dependencies(
        tag_state: Dict[str, Any],
        registry_entry: Dict[str, Any],
        bound_params: Dict[str, Any],
        get_instance_params: Callable[[str], Tuple[str, ...]],
    ) -> list:
        """
        Resolve structural dependencies from the current tag state.

        Find the last operation that wrote each referenced resource instance
        and resolve the implicit ``pipeline_config`` dependency.

        Parameters
        ----------
        tag_state : Dict[str, Any]
            Current tag state mapping concrete tags to instance ids.
        registry_entry : Dict[str, Any]
            Registry entry of the operation being logged.
        bound_params : Dict[str, Any]
            Fully resolved parameters of the call, used to identify which
            instance of each resource type is referenced.
        get_instance_params : Callable[[str], List[str]]
            Function to retrieve candidate parameter names for the resource
            instance.

        Returns
        -------
        list
            List of operation ids this entry directly depends on (deduped,
            order-preserving).
        """
        depends_on = []
        tags = list(registry_entry.get("affected_by_tags", [])) + list(
            DepsUtils._IMPLICIT_AFFECTED_BY_TAGS
        )
        for tag in tags:
            concrete_tags, _unresolved = DepsUtils._resolve_tag_template(
                tag, bound_params
            )
            for concrete_tag in concrete_tags:
                instances = tag_state.get(concrete_tag)
                if not instances:
                    continue
                instance_key = DepsUtils._extract_instance_key(
                    concrete_tag,bound_params, get_instance_params
                )
                for key in DepsUtils._instance_keys(instance_key):
                    dependency_id = instances.get(key)
                    if dependency_id is not None and dependency_id not in depends_on:
                        depends_on.append(dependency_id)
        return depends_on

    @staticmethod
    def update_tag_state(
        tag_state: Dict[str, Any],
        registry_entry: Dict[str, Any],
        bound_params: Dict[str, Any],
        entry_id: str,
        get_instance_params: Callable[[str], List[str]],
    ) -> Dict[str, Any]:
        """
        Mark this entry as the latest writer of every resource it emits.

        Parameters
        ----------
        tag_state : Dict[str, Any]
            Current tag state mapping concrete tags to instance ids.
        registry_entry : Dict[str, Any]
            Registry entry of the operation just logged.
        bound_params : Dict[str, Any]
            Fully resolved parameters of the call, used to identify which
            instance of each emitted resource type was written.
        entry_id : str
            The id of the entry just logged.

        Returns
        -------
        Dict[str, Any]
            Updates the tag state in-place and returns it.
        """
        for tag in registry_entry.get("emits_tags", []):
            concrete_tags, _unresolved = DepsUtils._resolve_tag_template(
                tag, bound_params
            )
            for concrete_tag in concrete_tags:
                instance_key = DepsUtils._extract_instance_key(
                    concrete_tag, bound_params, get_instance_params
                )
                for key in DepsUtils._instance_keys(instance_key):
                    tag_state.setdefault(concrete_tag, {})[key] = entry_id
        return tag_state

    @staticmethod
    def apply_resets(
        tag_state: Dict[str, Any],
        registry_entry: Dict[str, Any],
        bound_params: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Invalidate every instance of the resource type(s) this entry resets.

        Unlike ``emits_tags`` (which adds one instance to a resource type's
        tag_state), ``resets_tags`` clears the *entire* tag_state entry for a
        resource type - so a later operation that still references an
        instance name that existed before the reset correctly finds no
        producer, instead of the stale pre-reset one.

        Parameters
        ----------
        tag_state : Dict[str, Any]
            Current tag state mapping concrete tags to instance ids.
        registry_entry : Dict[str, Any]
            Registry entry of the operation just logged.
        bound_params : Dict[str, Any]
            Fully resolved parameters of the call.

        Returns
        -------
        Dict[str, Any]
            Updates the tag state in-place and returns it.
        """
        for tag in registry_entry.get("resets_tags", []):
            concrete_tags, unresolved = DepsUtils._resolve_tag_template(
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
        return tag_state
