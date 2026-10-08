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

"""Spec manager: single public access point for spec.json."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, Union, List, Dict

from ..entities.spec_data import SpecData

from ..services.spec_instances_service import SpecInstancesService
from ..services.spec_study_service import SpecStudyService

from ..helper.graph_helper import GraphHelper
from ..helper.spec_builder_helper import SpecBuilderHelper
from ..helper.spec_io_helper import SpecIOHelper
from ..helper.spec_validator_helper import SpecValidatorHelper

from ...utils.deps_utils import DepsUtils
from ...utils.operation_registry_utils import OperationRegistryUtils


class SpecManager:
    """
    Manager owning one SpecData instance for the one pipeline/project spec.json.

    Examples
    --------
    Manual construction:

    >>> spec_manager = SpecManager()
    >>> spec_manager.add("FeatureSelectorManager.create", name="key_features")
    'key_features'
    >>> spec_manager.add(
    ...     "DBSCANAddService.dbscan", selection_name="key_features", eps=0.5
    ... )
    'dbscan'

    From a real pipeline:

    >>> spec_manager = SpecManager()
    >>> spec_manager.read_from_pipeline(pipeline)
    >>> spec_manager.write_json("spec.json")
    """

    def __init__(self) -> None:
        """
        Initialize an empty spec, owning a fresh `SpecData` instance.

        Returns
        -------
        None
        """
        self.data = SpecData()
        self.validator = SpecValidatorHelper()

    def read_from_pipeline(self, pipeline: Any) -> None:
        """
        Import every operation logged by a real pipeline run into this spec.

        Walks ``pipeline.data.log["operations"]`` in dependency-layer order
        (see ``LogGraphHelper.layers``) and replays each operation through
        ``SpecManager.add``, so the result is structurally identical to
        building the same spec manually via ``SpecManager.add``/
        ``SpecInstancesService.add``.

        ``pipeline.data.log`` has the shape::

        >>>    log = {
        >>>        "operations": {
        >>>            entry_id: {
        >>>                "id": entry_id,
        >>>                "global_seq": global_seq,
        >>>                "type": operation_type,
        >>>                "config": config,
        >>>                "depends_on": [tags],
        >>>            },
        >>>            ...
        >>>        },
        >>>        "counters": {},
        >>>        "tag_state": {},
        >>>        "global_seq": 0,
        >>>    }

        where ``operation_type`` resolves (via `OperationRegistryUtils`) to a
        registry entry of the form::

        >>>    {
        >>>        "domain": domain_key,
        >>>        "emits_tags": [...],
        >>>        "affected_by_tags": [...],
        >>>        "resets_tags": [...],
        >>>        "technical_params": {...},
        >>>        "dispatch": {
        >>>            "module": ...,
        >>>            "class": ...,
        >>>            "method_name": ...,
        >>>        },
        >>>    }

        Parameters
        ----------
        pipeline : Any
            A pipeline object exposing ``pipeline.data.log`` in the above
            shape (e.g. `PipelineManager`).

        Returns
        -------
        None
        """
        log = pipeline.data.log  # dict
        graph = GraphHelper.build_graph(
            log["operations"], reduce=True
        )  # nx.Digraph
        for layer in GraphHelper.layers(graph):
            for entry_id in layer:
                operation = log["operations"][entry_id]
                operation_type = operation["type"]
                config = operation["config"]
                domain = OperationRegistryUtils.get_domain(operation_type)
                entry = OperationRegistryUtils.get_registry_entry(
                    operation_type
                )
                instance_name = SpecBuilderHelper.resolve_instance_name(
                    target=getattr(self.data, domain),
                    method_name=operation_type.rsplit(".", 1)[-1],
                    name=None,
                    instance_params=OperationRegistryUtils.get_instance_params(
                        domain
                    ),
                    kwargs=config,
                )
                params = SpecBuilderHelper.write_params(
                    config, entry.get("technical_params", {})
                )
                self._add_without_resync(
                    domain, instance_name, operation_type, params
                )
        # one resync for the whole import instead of one per operation
        SpecBuilderHelper.resync(self.data)

    def write_json(self, path: Union[str, Path]) -> None:
        """
        Write the current spec to a spec.json file.

        Parameters
        ----------
        path : str or Path
            Destination file path.

        Returns
        -------
        None
        """
        SpecIOHelper.write(self.data, path)

    def read_json(self, path: Union[str, Path]) -> None:
        """
        Replace the current spec with the contents of a spec.json file.

        Parameters
        ----------
        path : str or Path
            Source file path.

        Returns
        -------
        None
        """
        self.data = SpecIOHelper.read(path)

    def _add_without_resync(
        self,
        domain: str,
        instance_name: str,
        operation_type: Optional[str] = None,
        params: Optional[Dict[str, Any]] = None,
        depends_on: Optional[List[str]] = None,
    ) -> Optional[str]:
        """
        Add an instance (if needed) and append one modifier, without resync.

        Internal counterpart to `add` that skips the ``SpecBuilderHelper.
        resync`` call, so `read_from_pipeline` can import a whole log with a
        single resync at the end instead of one per operation.

        Parameters
        ----------
        domain : str
            Domain/module name where the instance is located.
        instance_name : str
            Name of the instance to add to (created if it does not exist
            yet).
        operation_type : str, optional
            Registered operation type of the call. If None, only ensures
            the (empty) instance exists and returns None.
        params : Dict[str, Any], optional
            Resolved config for the new modifier (see
            ``SpecBuilderHelper.write_params``).
        depends_on : List[str], optional
            Explicit dependency override for the new modifier.

        Returns
        -------
        str or None
            The new modifier's name (``mod_name``), or None if only an
            empty instance was created.
        """
        if operation_type is None:
            if self.data.get_instance(instance_name, domain) is None:
                self.data.add_instance(instance_name, domain, {"modifiers": []})
            return None

        # mod_id doubles as entry_id, mirroring the online
        # pipeline_data.log["counters"] scheme
        self.data.comfort_mode["counters"], mod_id = DepsUtils.next_counter(
            self.data.comfort_mode["counters"], operation_type
        )
        modifier = {
            "mod_name": mod_id,
            "type": operation_type,
            "config": params,
            "depends_on": list(depends_on) if depends_on is not None else [],
            "depends_on_override": depends_on is not None,
            "order": SpecBuilderHelper.next_order(self.data),
        }
        if self.data.get_instance(instance_name, domain) is None:
            self.data.add_instance(
                instance_name, domain, {"modifiers": [modifier]}
            )
        else:
            self.data.add_modifier(instance_name, domain, modifier)
        return mod_id

    def add(
        self,
        domain: str,
        instance_name: str,
        operation_type: Optional[str] = None,
        params: Optional[Dict[str, Any]] = None,
        depends_on: Optional[List[str]] = None,
    ) -> Optional[str]:
        """
        Add an instance (if needed) and append one modifier.

        Parameters
        ----------
        domain : str
            Domain/module name where the instance is located.
        instance_name : str
            Name of the instance to add to (created if it does not exist
            yet).
        operation_type : str, optional
            Registered operation type of the call. If None, only ensures
            the (empty) instance exists and returns None.
        params : Dict[str, Any], optional
            Resolved config for the new modifier (see
            ``SpecBuilderHelper.write_params``).
        depends_on : List[str], optional
            Explicit dependency override for the new modifier.

        Returns
        -------
        str or None
            The new modifier's name (``mod_name``), or None if only an
            empty instance was created.
        """
        mod_id = self._add_without_resync(
            domain, instance_name, operation_type, params, depends_on
        )
        SpecBuilderHelper.resync(self.data)
        return mod_id

    def update_modifier(
        self,
        domain: str,
        instance_name: str,
        mod_name: str,
        config: Optional[Dict[str, Any]] = None,
        depends_on: Optional[List[str]] = None,
    ) -> None:
        """
        Update a modifier's config and/or explicit ``depends_on``.

        Parameters
        ----------
        domain : str
            Domain/module name where the instance is located.
        instance_name : str
            Name of the instance holding the modifier.
        mod_name : str
            Name of the modifier to update.
        config : Dict[str, Any], optional
            Config values merged into the modifier's existing config.
        depends_on : List[str], optional
            Explicit dependency override for the modifier.

        Returns
        -------
        None
        """
        self.data.update_modifier(
            domain, instance_name, mod_name, config, depends_on
        )
        SpecBuilderHelper.resync(self.data)

    def set_modifier_depends_on(
        self,
        domain: str,
        instance_name: str,
        mod_name: str,
        depends_on: Optional[List[str]],
    ) -> None:
        """
        Set or clear a modifier's explicit ``depends_on`` override.

        Parameters
        ----------
        domain : str
            Domain/module name where the instance is located.
        instance_name : str
            Name of the instance holding the modifier.
        mod_name : str
            Name of the modifier to update.
        depends_on : List[str] or None
            New explicit dependency list, or None to clear the override
            and fall back to automatic resolution.

        Returns
        -------
        None
        """
        self.data.set_modifier_depends_on(
            domain, instance_name, mod_name, depends_on
        )
        SpecBuilderHelper.resync(self.data)

    def remove_modifier(
        self, domain: str, instance_name: str, mod_name: str
    ) -> None:
        """
        Remove a single modifier from an instance.

        Parameters
        ----------
        domain : str
            Domain/module name where the instance is located.
        instance_name : str
            Name of the instance holding the modifier.
        mod_name : str
            Name of the modifier to remove.

        Returns
        -------
        None
        """
        self.data.remove_modifier(domain, instance_name, mod_name)
        SpecBuilderHelper.resync(self.data)

    def reorder_modifier(
        self,
        domain: str,
        instance_name: str,
        mod_name: str,
        new_position: int,
    ) -> None:
        """
        Move a modifier to ``new_position`` among its instance's siblings.

        Parameters
        ----------
        domain : str
            Domain/module name where the instance is located.
        instance_name : str
            Name of the instance holding the modifier.
        mod_name : str
            Name of the modifier to reposition.
        new_position : int
            Target 0-based index among the instance's other modifiers.

        Returns
        -------
        None
        """
        SpecBuilderHelper.reorder(
            self.data, domain, instance_name, mod_name, new_position
        )

    def remove_instance(
        self,
        domain: str,
        instance_name: str,
    ) -> None:
        """
        Remove an instance together with all of its modifiers.

        Parameters
        ----------
        domain : str
            Domain/module name where the instance is located.
        instance_name : str
            Name of the instance to remove.

        Returns
        -------
        None
        """
        self.data.remove_instance(instance_name, domain)
        SpecBuilderHelper.resync(self.data)

    def list(
        self,
        domains: List[str] = None,
    ):
        """
        List instances across the given module domains.
        TODO: Not implemented yet.

        Parameters
        ----------
        domains : List[str], optional
            Module domain names to list instances from.
        """
        for domain in domains:
            print(domain)

    @property
    def instances(self) -> SpecInstancesService:
        """Instance-level facade over this manager's spec data."""
        return SpecInstancesService(self, self.data)

    @property
    def studies(self) -> SpecStudyService:
        """Study-level facade over this manager's spec data."""
        return SpecStudyService(self.data)

    def validate(self) -> None:
        """
        Validate every modifier's required params and cross-references.

        Delegates to ``SpecValidatorHelper.check_required_params``/
        ``check_cross_references`` for every modifier in every module, plus
        the required-param check for the ``pipeline`` singleton.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If any modifier is missing a required param or references a
            not-(yet)-existing instance in another module.
        """
        for module_name in self.data.MODULES:
            for instance in getattr(self.data, module_name).values():
                for modifier in instance["modifiers"]:
                    entry = OperationRegistryUtils.get_entry(modifier["type"])
                    SpecValidatorHelper.check_required_params(
                        modifier["type"], entry, modifier["config"]
                    )
                    SpecValidatorHelper.check_cross_references(
                        self.data, entry, module_name, modifier["config"]
                    )
        for modifier in self.data.pipeline["modifiers"]:
            entry = OperationRegistryUtils.get_entry(modifier["type"])
            SpecValidatorHelper.check_required_params(
                modifier["type"], entry, modifier["config"]
            )

    def print_info(self):
        """Print a human-readable summary of every module/instance/modifier."""
        print("SpecManager Information:")
        print("Modules:")
        for module_name in self.data.MODULES:
            print(f"  {module_name}")
            for instance_name, instance in getattr(
                self.data, module_name
            ).items():
                print(f"    Instance: {instance_name}")
                if (
                    len(instance["modifiers"]) > 0
                    and not instance["modifiers"][0] == {}
                ):
                    for modifier in instance["modifiers"]:
                        print(f"      Modifier: {modifier['mod_name']}")
                        for key, value in modifier["config"].items():
                            print(f"        {key}: {value}")
                        print("      Depends on:")
                        if modifier.get("depends_on"):
                            for dependency in modifier.get("depends_on", []):
                                print(f"            {dependency}")
