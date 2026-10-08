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
Spec data container for the one pipeline/project spec.json.

This module contains the SpecData class, the registry-driven replacement for
the old VOR-field-snapshot spec.json. See plan.md / repo memory
spec_module_design.md for the full design rationale.
"""

from typing import Any, Dict, List, Optional, Union


class SpecData:
    """
    Pure data container for the one pipeline/project spec.json.

    Analogous to PipelineData: only ever populated/mutated through
    SpecManager, never directly. Every module is a `Dict[name, instance]`,
    where an instance is uniformly `{"modifiers": [{"type": operation_type,
    "config": {...}}, ...]}` - no separate base-config field, even the
    creating call is just the first modifier entry. `pipeline` is the one
    exception: a singleton (no name dict) since there is only ever one per
    pipeline.

    Attributes
    ----------
    pipeline : Dict[str, dict]
        Pipeline instances keyed by name (`pipeline_init`/`update_config`/
        `clear_all`/`add_custom_metadata` calls per instance, in order) -
        a normal instance dict like every other module, not a singleton
    trajectory : Dict[str, dict]
        Trajectory instances keyed by name (branching allowed via modifiers
        referencing another instance as source)
    feature : Dict[str, dict]
        Feature instances keyed by name
    feature_selection : Dict[str, dict]
        Feature selector instances keyed by name
    decomposition : Dict[str, dict]
        Decomposition instances keyed by name
    clustering : Dict[str, dict]
        Clustering instances keyed by name
    data_selector : Dict[str, dict]
        Data selector instances keyed by name
    comparison : Dict[str, dict]
        Comparison instances keyed by name
    feature_importance : Dict[str, dict]
        Feature importance instances keyed by name
    structure_visualization : Dict[str, dict]
        Structure visualization instances keyed by name
    plots : Dict[str, dict]
        Plot instances keyed by name (default names, since plot calls don't
        take an explicit name parameter)
    analysis : Dict[str, dict]
        Analysis instances keyed by name (RMSD/RMSF/... services, default
        names for the same reason as `plots`)
    studies : Dict[str, Dict[str, Dict[str, Union[str, List[str]]]]]
        Optional named reference bundles:
        `{study_name: {domain: {instance_name: "all" | [mod_name, ...]}}}`.
        ``"all"`` is resolved dynamically at use time (not a snapshot), so
        it tracks modifiers added to the instance after the study was
        defined.
    """

    MODULES = (
        "pipeline",
        "trajectory",
        "feature",
        "feature_selection",
        "decomposition",
        "clustering",
        "data_selector",
        "comparison",
        "feature_importance",
        "structure_visualization",
        "plots",
        "analysis",
        "studies",
    )

    def __init__(self) -> None:
        """
        Initialize an empty spec data container.

        Metadata for each module initialized as empty dicts:
            pipeline,
            trajectory,
            feature,
            feature_selection,
            decomposition,
            clustering,
            data_selector,
            comparison,
            feature_importance,
            analysis,
            structure_visualization,
            plots,
            studies,

        Besides the module dicts, two more attributes are initialized:
            comfort_mode : dict
                Runtime-only bookkeeping, never written to spec.json (see
                ``SpecIOHelper``), analogous to ``pipeline_data.log``'s
                ``counters``/``tag_state`` for the real pipeline:

                - ``tag_state``: resolved tag state after replaying every
                  modifier in ``order`` (see ``SpecBuilderHelper.resync``),
                  used to auto-resolve new modifiers' ``depends_on``.
                - ``order_counter``: last assigned global modifier
                  ``order`` value (see ``SpecBuilderHelper.next_order``).
                - ``counters``: per-operation-type sequence counters used
                  to derive default ``mod_name``s when importing from a
                  real pipeline (see ``SpecManager.read_from_pipeline``).
            graph : nx.DiGraph or None
                Operation dependency graph rebuilt on every
                ``SpecBuilderHelper.resync`` call (see
                ``LogGraphHelper.build_graph``); None until the first
                resync.
            replay_results : dict
                Runtime-only, never written to spec.json: return values of
                the most recent ``SpecManager.new_pipeline()`` replay (see
                ``PipelineBuilderHelper``), keyed by the modifier's
                ``mod_name``. Overwritten on every replay (holds only the
                latest build, no history) - the intended usage is a single
                build/run per process (e.g. a CLI job on an HPC cluster),
                not repeated builds from the same spec.

        Returns
        -------
        None
        """
        for module_name in self.MODULES:
            setattr(self, module_name, {})

        self.comfort_mode = {
            "tag_state": {},
            "order_counter": 0,
            "counters": {},
        }
        self.graph = None
        self.replay_results: Dict[str, Any] = {}

    def add_instance(
        self, instance_name: str, domain: str, instance: dict
    ) -> None:
        """
        Add a new instance to the specified domain.

        Parameters
        ----------
        instance_name : str
            Name of the instance to add.
        domain : str
            Domain/module name where the instance should be added.
        instance : dict
            The instance dictionary to add.

        Returns
        -------
        None
        """
        getattr(self, domain)[instance_name] = instance

    def add_modifier(
        self, instance_name: str, domain: str, modifier: dict
    ) -> None:
        """
        Add a modifier to an existing instance in the specified domain.

        Parameters
        ----------
        instance_name : str
            Name of the instance to which the modifier should be added.
        domain : str
            Domain/module name where the instance is located.
        modifier : dict
            The modifier dictionary to add.

        Returns
        -------
        None
        """
        instance = self.get_instance(instance_name, domain)
        if instance is not None:
            instance["modifiers"].append(modifier)

    def get_instance(self, instance_name: str, domain: str) -> dict:
        """
        Retrieve an instance by name from the specified domain.

        Parameters
        ----------
        instance_name : str
            Name of the instance to retrieve.
        domain : str
            Domain/module name where the instance is located.

        Returns
        -------
        dict
            The instance dictionary if found, otherwise None.
        """
        return getattr(self, domain).get(instance_name, None)

    def remove_instance(self, instance_name: str, domain: str) -> None:
        """
        Remove an instance together with all of its modifiers.

        Parameters
        ----------
        instance_name : str
            Name of the instance to remove.
        domain : str
            Domain/module name where the instance is located.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If the instance does not exist.
        """
        if self.get_instance(instance_name, domain) is None:
            raise ValueError(
                f"Instance '{instance_name}' not found in domain '{domain}'."
            )
        del getattr(self, domain)[instance_name]

    def get_modifier(
        self, domain: str, instance_name: str, mod_name: str
    ) -> dict:
        """
        Retrieve a single modifier from an instance by its ``mod_name``.

        Parameters
        ----------
        domain : str
            Domain/module name where the instance is located.
        instance_name : str
            Name of the instance holding the modifier.
        mod_name : str
            Name of the modifier to retrieve.

        Returns
        -------
        dict
            The modifier dictionary.

        Raises
        ------
        ValueError
            If the instance or the modifier does not exist.
        """
        instance = self.get_instance(instance_name, domain)
        if instance is None:
            raise ValueError(
                f"Instance '{instance_name}' not found in domain '{domain}'."
            )
        for modifier in instance["modifiers"]:
            if modifier.get("mod_name") == mod_name:
                return modifier
        raise ValueError(
            f"Modifier '{mod_name}' not found in instance "
            f"'{instance_name}' of domain '{domain}'."
        )

    def list_modifiers(self, domain: str, instance_name: str) -> list:
        """
        List all modifiers of a single instance, in storage order.

        Parameters
        ----------
        domain : str
            Domain/module name where the instance is located.
        instance_name : str
            Name of the instance whose modifiers should be listed.

        Returns
        -------
        list
            Copy of the instance's modifier list.

        Raises
        ------
        ValueError
            If the instance does not exist.
        """
        instance = self.get_instance(instance_name, domain)
        if instance is None:
            raise ValueError(
                f"Instance '{instance_name}' not found in domain '{domain}'."
            )
        return list(instance["modifiers"])

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
            Config values merged (via ``dict.update``) into the modifier's
            existing config.
        depends_on : List[str], optional
            If given, overrides the modifier's dependencies and marks them
            as explicit (``depends_on_override``), so they no longer get
            auto-resolved on the next ``SpecBuilderHelper.resync``.

        Returns
        -------
        None
        """
        modifier = self.get_modifier(domain, instance_name, mod_name)
        if config:
            modifier["config"].update(config)
        if depends_on is not None:
            modifier["depends_on"] = list(depends_on)
            modifier["depends_on_override"] = True

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
            New explicit dependency list, or None to clear the override and
            fall back to automatic resolution on the next
            ``SpecBuilderHelper.resync``.

        Returns
        -------
        None
        """
        modifier = self.get_modifier(domain, instance_name, mod_name)
        modifier["depends_on_override"] = depends_on is not None
        if depends_on is not None:
            modifier["depends_on"] = list(depends_on)

    def remove_modifier(
        self, domain: str, instance_name: str, mod_name: str
    ) -> None:
        """
        Remove a single modifier from an instance.

        If the removed modifier was the instance's last remaining one, the
        whole instance is removed as well (an instance with no modifiers
        has no creating call left and is therefore meaningless).

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
        modifier = self.get_modifier(domain, instance_name, mod_name)
        instance = self.get_instance(instance_name, domain)
        instance["modifiers"].remove(modifier)
        if not instance["modifiers"]:
            self.remove_instance(instance_name, domain)

    def get_all_modifiers(
        self,
        instance_name: Optional[str] = None,
        domain: Optional[str] = None,
    ) -> list:
        """
        Retrieve all modifiers for a given instance in the specified domain.
        If no instance_name is provided, retrieves modifiers for all
        instances in the domain. If no domain is provided, retrieves
        modifiers for all instances across all domains.

        Parameters
        ----------
        instance_name : str
            Name of the instance whose modifiers should be retrieved.
        domain : str
            Domain/module name where the instance is located.

        Returns
        -------
        list
            List of modifier dictionaries for the instance, or an empty
            list if the instance is not found.
        """
        domain_dicts = {}
        if domain is None:
            for domain_name in self.MODULES:
                domain_dicts[domain_name] = getattr(self, domain_name)
        else:
            domain_dicts[domain] = getattr(self, domain)

        instance_dicts = {}
        if instance_name is None:
            for domain_name, domain_instances in domain_dicts.items():
                for inst_name, inst in domain_instances.items():
                    instance_dicts[inst_name] = inst
        else:
            for domain_name, domain_instances in domain_dicts.items():
                if instance_name in domain_instances:
                    instance_dicts[instance_name] = domain_instances[
                        instance_name
                    ]

        modifiers = []
        for inst_name, inst in instance_dicts.items():
            modifiers.extend(inst.get("modifiers", []))

        return modifiers

    def add_new_study(self, study_name: str, exist_ok: bool = False) -> None:
        """
        Create a new, empty named study.

        Parameters
        ----------
        study_name : str
            Name of the study to create.
        exist_ok : bool, default=False
            If False, raise when a study of this name already exists.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If ``study_name`` already exists and ``exist_ok`` is False.
        """
        if study_name in self.studies and not exist_ok:
            raise ValueError(f"Study '{study_name}' already exists.")
        self.studies.setdefault(study_name, {})

    def remove_study(self, study_name: str) -> None:
        """
        Remove an entire named study.

        Parameters
        ----------
        study_name : str
            Name of the study to remove.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If ``study_name`` does not exist.
        """
        if study_name not in self.studies:
            raise ValueError(f"Study '{study_name}' not found.")
        del self.studies[study_name]

    def add_to_study(
        self,
        study_name: str,
        domain: str,
        instance_name: str,
        mod_names: Union[str, List[str]] = "all",
    ) -> None:
        """
        Add or overwrite one instance reference within a study.

        Creates the study and/or the domain group within it if they do not
        exist yet (upsert).

        Parameters
        ----------
        study_name : str
            Name of the study to add to.
        domain : str
            Domain/module name the referenced instance belongs to.
        instance_name : str
            Name of the referenced instance.
        mod_names : str or List[str], default="all"
            Either the literal ``"all"`` (every modifier of the instance,
            resolved dynamically at use time) or an explicit list of
            ``mod_name``s to reference.

        Returns
        -------
        None
        """
        study = self.studies.setdefault(study_name, {})
        study.setdefault(domain, {})[instance_name] = mod_names

    def update_study(
        self,
        study_name: str,
        domain: str,
        instance_name: str,
        mod_names: Union[str, List[str]],
    ) -> None:
        """
        Update an existing instance reference within a study.

        Unlike ``add_to_study``, this requires the study/domain/instance
        reference to already exist (no upsert).

        Parameters
        ----------
        study_name : str
            Name of the study to update.
        domain : str
            Domain/module name the referenced instance belongs to.
        instance_name : str
            Name of the referenced instance.
        mod_names : str or List[str]
            New value, either the literal ``"all"`` or an explicit list of
            ``mod_name``s.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If the study, domain group, or instance reference does not
            exist yet.
        """
        if study_name not in self.studies:
            raise ValueError(f"Study '{study_name}' not found.")
        domain_group = self.studies[study_name].get(domain)
        if domain_group is None or instance_name not in domain_group:
            raise ValueError(
                f"Reference '{domain}.{instance_name}' not found in study "
                f"'{study_name}'."
            )
        domain_group[instance_name] = mod_names

    def remove_from_study(
        self,
        study_name: str,
        domain: str,
        instance_name: Optional[str] = None,
    ) -> None:
        """
        Remove one instance reference, or a whole domain group, from a study.

        The study itself is left in place even if this empties it entirely
        (use ``remove_study`` to delete the study itself).

        Parameters
        ----------
        study_name : str
            Name of the study to remove from.
        domain : str
            Domain/module name to remove from, or to remove an instance
            reference within.
        instance_name : str, optional
            If given, only this instance reference is removed (the domain
            group is dropped as well if it becomes empty). If None, the
            whole domain group is removed.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If the study, domain group, or (if given) instance reference
            does not exist.
        """
        if study_name not in self.studies:
            raise ValueError(f"Study '{study_name}' not found.")
        domain_group = self.studies[study_name].get(domain)
        if domain_group is None:
            raise ValueError(
                f"Domain '{domain}' not found in study '{study_name}'."
            )
        if instance_name is None:
            del self.studies[study_name][domain]
            return
        if instance_name not in domain_group:
            raise ValueError(
                f"Reference '{domain}.{instance_name}' not found in study "
                f"'{study_name}'."
            )
        del domain_group[instance_name]
        if not domain_group:
            del self.studies[study_name][domain]

    def get_study(
        self,
        study_name: str,
        domain: Optional[str] = None,
        instance_name: Optional[str] = None,
    ) -> Any:
        """
        Read a study, a domain group within it, or a single reference value.

        Parameters
        ----------
        study_name : str
            Name of the study to read from.
        domain : str, optional
            If given, scope the read to this domain group.
        instance_name : str, optional
            If given (requires ``domain``), scope the read to this single
            instance's ``mod_names`` value.

        Returns
        -------
        Any
            The whole study dict, the domain group dict, or the single
            ``mod_names`` value, depending on how far ``domain``/
            ``instance_name`` are given.

        Raises
        ------
        ValueError
            If ``study_name``/``domain``/``instance_name`` does not exist,
            or if ``instance_name`` is given without ``domain``.
        """
        if study_name not in self.studies:
            raise ValueError(f"Study '{study_name}' not found.")
        study = self.studies[study_name]
        if domain is None:
            if instance_name is not None:
                raise ValueError("instance_name requires domain to be set.")
            return study
        if domain not in study:
            raise ValueError(
                f"Domain '{domain}' not found in study '{study_name}'."
            )
        domain_group = study[domain]
        if instance_name is None:
            return domain_group
        if instance_name not in domain_group:
            raise ValueError(
                f"Reference '{domain}.{instance_name}' not found in study "
                f"'{study_name}'."
            )
        return domain_group[instance_name]
