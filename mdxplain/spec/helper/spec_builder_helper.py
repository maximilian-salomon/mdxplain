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

"""Core instance/modifier mutation logic backing `SpecManager`."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING
from ...utils.deps_utils import DepsUtils
from ...utils.operation_registry_utils import OperationRegistryUtils
from .graph_helper import GraphHelper

if TYPE_CHECKING:
    from ..entities.spec_data import SpecData


class SpecBuilderHelper:
    """
    Static helper implementing the core mutation logic behind `SpecManager`.

    Covers name resolution for new instances/modifiers, the global
    modifier ``order``, auto-resolution of ``depends_on`` plus the
    operation graph rebuild (``resync``), modifier reordering, and
    resolving a call's config against its registered ``technical_params``
    (``write_params``).
    """

    @staticmethod
    def resolve_instance_name(
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
    def resolve_modifier_name(
        targets: list[Dict[str, Any]],
        method_name: str,
    ) -> str:
        """
        Resolve a fresh, globally unique modifier name for ``method_name``.

        Parameters
        ----------
        targets : list[Dict[str, Any]]
            One or more modules' `Dict[name, instance]`, searched for
            already-used ``mod_name`` values to avoid collisions.
        method_name : str
            Method name (the part of `operation_type` after the last ".").

        Returns
        -------
        str
            A `"{method_name}_{n}"` name not yet used by any modifier in
            ``targets``.
        """
        existing_names = {
            modifier.get("mod_name")
            for target in targets
            for instance in target.values()
            if isinstance(instance, dict)
            for modifier in instance.get("modifiers", [])
        }

        n = 1
        while f"{method_name}_{n}" in existing_names:
            n += 1

        return f"{method_name}_{n}"

    @staticmethod
    def next_order(spec_data: SpecData) -> int:
        """
        Advance and return the global modifier order counter.

        Parameters
        ----------
        spec_data : SpecData
            The spec data container whose order counter is advanced.

        Returns
        -------
        int
            The new counter value, to be stored as a modifier's ``order``.
        """
        spec_data.comfort_mode["order_counter"] += 1
        return spec_data.comfort_mode["order_counter"]

    @staticmethod
    def _ordered_modifiers(spec_data: SpecData) -> List[Dict[str, Any]]:
        """
        Collect every modifier across all modules, sorted by ``order``.

        Backfills a missing ``order`` field (for specs read from a JSON
        written before the field existed) by appending such modifiers
        after the highest existing order, in encounter order.

        Parameters
        ----------
        spec_data : SpecData
            The spec data container to collect modifiers from.

        Returns
        -------
        List[Dict[str, Any]]
            All modifiers (``studies`` excluded), sorted by ``order``.
        """
        modifiers = []
        for domain in spec_data.MODULES:
            if domain == "studies":
                continue
            for instance in getattr(spec_data, domain).values():
                modifiers.extend(instance.get("modifiers", []))

        # specs read from json may predate the order field
        last_order = max(
            (m["order"] for m in modifiers if "order" in m), default=0
        )
        for modifier in modifiers:
            if "order" not in modifier:
                last_order += 1
                modifier["order"] = last_order
        spec_data.comfort_mode["order_counter"] = max(
            spec_data.comfort_mode["order_counter"], last_order
        )
        return sorted(modifiers, key=lambda m: m["order"])

    @staticmethod
    def resync(spec_data: SpecData) -> None:
        """
        Recompute auto-resolved ``depends_on`` and rebuild the operation graph.

        Replays every modifier (in ``order``) through the tag-state
        resolution used at real pipeline runtime, so GUI-/manually-built
        specs get the same ``depends_on`` auto-resolution without executing
        anything. Modifiers with an explicit ``depends_on_override`` keep
        their stored dependencies instead of being recomputed. Must be
        called after any mutation that can affect dependency resolution
        (adding/removing/reordering modifiers).

        Parameters
        ----------
        spec_data : SpecData
            The spec data container to resync. Mutated in place: modifier
            ``depends_on`` values, each instance's modifier order, and
            ``spec_data.graph``.

        Returns
        -------
        None
        """
        modifiers = SpecBuilderHelper._ordered_modifiers(spec_data)
        get_instance_params = OperationRegistryUtils.get_instance_params

        tag_state: Dict[str, Any] = {}
        operations: Dict[str, Any] = {}
        for modifier in modifiers:
            entry = OperationRegistryUtils.get_registry_entry(modifier["type"])
            config = modifier["config"]
            # json written before the flag existed keeps its stored depends_on
            if not modifier.setdefault("depends_on_override", True):
                modifier["depends_on"] = DepsUtils.resolve_dependencies(
                    tag_state, entry, config, get_instance_params
                )
            tag_state = DepsUtils.update_tag_state(
                tag_state,
                entry,
                config,
                modifier["mod_name"],
                get_instance_params,
            )
            tag_state = DepsUtils.apply_resets(tag_state, entry, config)
            operations[modifier["mod_name"]] = {
                "id": modifier["mod_name"],
                "type": modifier["type"],
                "config": config,
                "depends_on": modifier["depends_on"],
            }

        for domain in spec_data.MODULES:
            if domain == "studies":
                continue
            for instance in getattr(spec_data, domain).values():
                instance["modifiers"].sort(key=lambda m: m["order"])

        spec_data.comfort_mode["tag_state"] = tag_state
        spec_data.graph = GraphHelper.build_graph(operations, reduce=True)

    @staticmethod
    def reorder(
        spec_data: SpecData,
        domain: str,
        instance_name: str,
        mod_name: str,
        new_position: int,
    ) -> None:
        """
        Move a modifier to ``new_position`` among its instance's siblings.

        Reorders the modifier within its own instance, then re-numbers the
        global ``order`` of every modifier across all modules so its
        position relative to other instances' modifiers stays consistent,
        and finally triggers a resync.

        Parameters
        ----------
        spec_data : SpecData
            The spec data container to mutate.
        domain : str
            Domain/module name where the instance is located.
        instance_name : str
            Name of the instance holding the modifier.
        mod_name : str
            Name of the modifier to reposition.
        new_position : int
            Target 0-based index among the instance's other modifiers
            (clamped to the valid range).

        Returns
        -------
        None
        """
        target = spec_data.get_modifier(domain, instance_name, mod_name)
        siblings = sorted(
            (
                m
                for m in spec_data.get_instance(instance_name, domain)[
                    "modifiers"
                ]
                if m is not target
            ),
            key=lambda m: m["order"],
        )
        new_position = max(0, min(new_position, len(siblings)))

        sequence = SpecBuilderHelper._ordered_modifiers(spec_data)
        if siblings:
            sequence.remove(target)
            if new_position == 0:
                sequence.insert(sequence.index(siblings[0]), target)
            else:
                sequence.insert(
                    sequence.index(siblings[new_position - 1]) + 1, target
                )
            for order, modifier in enumerate(sequence, start=1):
                modifier["order"] = order
            spec_data.comfort_mode["order_counter"] = len(sequence)

        SpecBuilderHelper.resync(spec_data)

    @staticmethod
    def write_params(
        config: Dict[str, Any],
        technical_params: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Resolve a modifier's config against its registry-declared parameters.

        For every declared ``technical_param``, takes the value from
        ``config`` if present, otherwise falls back to its registered
        default. Required params missing from ``config`` raise instead of
        silently defaulting.

        Parameters
        ----------
        config : Dict[str, Any]
            The call's raw config values (e.g. from ``**kwargs``).
        technical_params : Dict[str, Any]
            Registry-declared params for this operation type, each with
            ``"required"`` (bool) and, if not required, ``"default"``.

        Returns
        -------
        Dict[str, Any]
            Config with one entry per declared param, defaults filled in.

        Raises
        ------
        ValueError
            If a required param is missing from ``config``.
        """
        new_params = {}
        for param in technical_params:
            if param not in config and technical_params[param].get(
                "required", True
            ):
                raise ValueError(
                    f"Missing required technical parameter: {param}"
                )
            if param in config:
                new_params[param] = config[param]
            else:
                new_params[param] = technical_params[param]["default"]
        return new_params
