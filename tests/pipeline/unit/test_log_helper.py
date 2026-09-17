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

"""Unit tests for instance-scoped dependency resolution in LogHelper."""

import networkx as nx
import pytest

from mdxplain.pipeline.entities.pipeline_data import PipelineData
from mdxplain.spec.helper.log_graph_helper import LogGraphHelper
from mdxplain.pipeline.helper.log_helper.log_helper import LogHelper
from mdxplain.utils.registry_utils import RegistryUtils


class _Owner:
    """Stand-in owner class for registry entries created inside the tests."""


def _register(operation_type, emits, affected_by, technical_params, resets=None):
    """Register a synthetic operation and return its dispatch method name."""
    method_name = operation_type
    RegistryUtils.register_operation(
        operation_type,
        {
            "dispatch": (_Owner, method_name),
            "emits_tags": emits,
            "affected_by_tags": affected_by,
            "resets_tags": resets or [],
            "technical_params": technical_params,
            "gui_param_info": {},
        },
    )
    return method_name


@pytest.fixture
def pipeline_data():
    """Provide a bare PipelineData with an empty operations log."""
    return PipelineData()


class TestSingletonResources:
    """Resources without an instance name resolve to their latest writer."""

    def test_repeated_mutation_resolves_to_latest(self, pipeline_data):
        """A consumer depends on the most recent mutation, not the first one."""
        mutate = _register("singleton_mutate", ["trajectory"], ["trajectory"], [])
        consume = _register("singleton_consume", [], ["trajectory"], [])

        LogHelper.log_operation(pipeline_data, _Owner, mutate, {})
        LogHelper.log_operation(pipeline_data, _Owner, mutate, {})
        LogHelper.log_operation(pipeline_data, _Owner, consume, {})

        operations = pipeline_data.log["operations"]
        assert operations["singleton_mutate_2"]["depends_on"] == ["singleton_mutate_1"]
        assert operations["singleton_consume_1"]["depends_on"] == ["singleton_mutate_2"]


class TestNamedInstances:
    """Named resources of the same type stay independent of each other."""

    def test_two_instances_do_not_collide(self, pipeline_data):
        """A consumer resolves the clustering it names, not the latest one."""
        produce = _register("cluster_add", ["clustering"], [], ["cluster_name"])
        consume = _register("cluster_use", [], ["clustering"], ["clustering_name"])

        LogHelper.log_operation(pipeline_data, _Owner, produce, {"cluster_name": "A"})
        LogHelper.log_operation(pipeline_data, _Owner, produce, {"cluster_name": "B"})
        LogHelper.log_operation(
            pipeline_data, _Owner, consume, {"clustering_name": "A"}
        )

        assert pipeline_data.log["operations"]["cluster_use_1"]["depends_on"] == [
            "cluster_add_1"
        ]

    def test_recreated_instance_replaces_previous(self, pipeline_data):
        """Reusing a name makes later consumers depend on the newer operation."""
        produce = _register("cluster_add", ["clustering"], [], ["cluster_name"])
        consume = _register("cluster_use", [], ["clustering"], ["clustering_name"])

        LogHelper.log_operation(pipeline_data, _Owner, produce, {"cluster_name": "A"})
        LogHelper.log_operation(pipeline_data, _Owner, produce, {"cluster_name": "A"})
        LogHelper.log_operation(
            pipeline_data, _Owner, consume, {"clustering_name": "A"}
        )

        assert pipeline_data.log["operations"]["cluster_use_1"]["depends_on"] == [
            "cluster_add_2"
        ]

    def test_list_valued_parameter_resolves_every_name(self, pipeline_data):
        """A parameter naming several instances yields one dependency each."""
        produce = _register("group_add", ["data_selector"], [], ["group_name"])
        consume = _register(
            "group_use", [], ["data_selector"], ["data_selector_groups"]
        )

        LogHelper.log_operation(pipeline_data, _Owner, produce, {"group_name": "g1"})
        LogHelper.log_operation(pipeline_data, _Owner, produce, {"group_name": "g2"})
        LogHelper.log_operation(
            pipeline_data, _Owner, consume, {"data_selector_groups": ["g1", "g2"]}
        )

        assert pipeline_data.log["operations"]["group_use_1"]["depends_on"] == [
            "group_add_1",
            "group_add_2",
        ]

    def test_unknown_instance_yields_no_dependency(self, pipeline_data):
        """Naming an instance that was never emitted produces no edge."""
        produce = _register("cluster_add", ["clustering"], [], ["cluster_name"])
        consume = _register("cluster_use", [], ["clustering"], ["clustering_name"])

        LogHelper.log_operation(pipeline_data, _Owner, produce, {"cluster_name": "A"})
        LogHelper.log_operation(
            pipeline_data, _Owner, consume, {"clustering_name": "B"}
        )

        assert pipeline_data.log["operations"]["cluster_use_1"]["depends_on"] == []

    def test_two_different_candidate_params_both_resolve(self, pipeline_data):
        """A call naming two distinct instances via different params gets both."""
        produce = _register(
            "selector_add", ["feature_selection"], [], ["selector_name"]
        )
        consume = _register(
            "structviz_create",
            [],
            ["feature_selection"],
            ["selector_centroid", "selector_features"],
        )

        LogHelper.log_operation(
            pipeline_data, _Owner, produce, {"selector_name": "coords_all"}
        )
        LogHelper.log_operation(
            pipeline_data, _Owner, produce, {"selector_name": "important_distances"}
        )
        LogHelper.log_operation(
            pipeline_data,
            _Owner,
            consume,
            {
                "selector_centroid": "coords_all",
                "selector_features": "important_distances",
            },
        )

        assert pipeline_data.log["operations"]["structviz_create_1"][
            "depends_on"
        ] == ["selector_add_1", "selector_add_2"]


class TestImplicitPipelineConfig:
    """Every operation implicitly depends on the current pipeline_config."""

    def test_operation_depends_on_prior_config(self, pipeline_data):
        """An unrelated operation still links back to a config-emitting one."""
        init = _register("config_init", ["pipeline_config"], [], [])
        consume = _register("config_consume", [], [], [])

        LogHelper.log_operation(pipeline_data, _Owner, init, {})
        LogHelper.log_operation(pipeline_data, _Owner, consume, {})

        assert pipeline_data.log["operations"]["config_consume_1"]["depends_on"] == [
            "config_init_1"
        ]

    def test_config_update_chains_to_previous_config(self, pipeline_data):
        """A second config-emitting call depends on the first, not itself."""
        init = _register("config_init", ["pipeline_config"], [], [])
        update = _register("config_update", ["pipeline_config"], [], [])

        LogHelper.log_operation(pipeline_data, _Owner, init, {})
        LogHelper.log_operation(pipeline_data, _Owner, update, {})

        assert pipeline_data.log["operations"]["config_update_1"]["depends_on"] == [
            "config_init_1"
        ]

    def test_no_dependency_before_any_config_emitted(self, pipeline_data):
        """Without a prior config emitter there is nothing to depend on."""
        consume = _register("config_consume", [], [], [])

        LogHelper.log_operation(pipeline_data, _Owner, consume, {})

        assert pipeline_data.log["operations"]["config_consume_1"]["depends_on"] == []


class TestResetsTags:
    """resets_tags invalidates a whole resource type instead of one instance."""

    def test_reset_clears_all_named_instances(self, pipeline_data):
        """After a reset, old instance names no longer resolve to anything."""
        produce = _register("cluster_add", ["clustering"], [], ["cluster_name"])
        reset = _register("cluster_reset", [], [], [], resets=["clustering"])
        consume = _register("cluster_use", [], ["clustering"], ["clustering_name"])

        LogHelper.log_operation(pipeline_data, _Owner, produce, {"cluster_name": "A"})
        LogHelper.log_operation(pipeline_data, _Owner, reset, {})
        LogHelper.log_operation(
            pipeline_data, _Owner, consume, {"clustering_name": "A"}
        )

        assert pipeline_data.log["operations"]["cluster_use_1"]["depends_on"] == []

    def test_templated_tag_resolves_dynamic_resource(self, pipeline_data):
        """A `{param}` placeholder names the resource type from its value."""
        produce = _register(
            "feature_add", ["feature.{feature_type}"], [], ["feature_type"]
        )
        consume = _register(
            "feature_use", [], ["feature.{feature_type}"], ["feature_type"]
        )

        LogHelper.log_operation(
            pipeline_data, _Owner, produce, {"feature_type": "distances"}
        )
        LogHelper.log_operation(
            pipeline_data, _Owner, consume, {"feature_type": "distances"}
        )
        LogHelper.log_operation(
            pipeline_data, _Owner, consume, {"feature_type": "contacts"}
        )

        operations = pipeline_data.log["operations"]
        assert operations["feature_use_1"]["depends_on"] == ["feature_add_1"]
        assert operations["feature_use_2"]["depends_on"] == []

    def test_templated_reset_with_none_clears_matching_prefix(self, pipeline_data):
        """A placeholder resolving to None resets every tag sharing its prefix."""
        produce = _register(
            "feature_add", ["feature.{feature_type}"], [], ["feature_type"]
        )
        reset_all = _register(
            "feature_reset", [], [], ["feature_type"], resets=["feature.{feature_type}"]
        )
        consume = _register(
            "feature_use", [], ["feature.{feature_type}"], ["feature_type"]
        )

        LogHelper.log_operation(
            pipeline_data, _Owner, produce, {"feature_type": "distances"}
        )
        LogHelper.log_operation(
            pipeline_data, _Owner, produce, {"feature_type": "contacts"}
        )
        LogHelper.log_operation(pipeline_data, _Owner, reset_all, {"feature_type": None})
        LogHelper.log_operation(
            pipeline_data, _Owner, consume, {"feature_type": "distances"}
        )

        assert pipeline_data.log["operations"]["feature_use_1"]["depends_on"] == []

    def test_templated_reset_with_value_clears_only_that_tag(self, pipeline_data):
        """A resolved placeholder value only resets that one concrete tag."""
        produce = _register(
            "feature_add", ["feature.{feature_type}"], [], ["feature_type"]
        )
        reset_one = _register(
            "feature_reset", [], [], ["feature_type"], resets=["feature.{feature_type}"]
        )
        consume = _register(
            "feature_use", [], ["feature.{feature_type}"], ["feature_type"]
        )

        LogHelper.log_operation(
            pipeline_data, _Owner, produce, {"feature_type": "distances"}
        )
        LogHelper.log_operation(
            pipeline_data, _Owner, produce, {"feature_type": "contacts"}
        )
        LogHelper.log_operation(
            pipeline_data, _Owner, reset_one, {"feature_type": "distances"}
        )
        LogHelper.log_operation(
            pipeline_data, _Owner, consume, {"feature_type": "distances"}
        )
        LogHelper.log_operation(
            pipeline_data, _Owner, consume, {"feature_type": "contacts"}
        )

        operations = pipeline_data.log["operations"]
        assert operations["feature_use_1"]["depends_on"] == []
        assert operations["feature_use_2"]["depends_on"] == ["feature_add_2"]


class TestGraphReduction:
    """Transitive reduction removes implied edges without losing reachability."""

    def test_reduction_keeps_reachability(self, pipeline_data):
        """Redundant shortcut edges disappear but every node stays reachable."""
        load = _register("graph_load", ["trajectory"], [], [])
        mutate = _register("graph_mutate", ["trajectory"], ["trajectory"], [])
        consume = _register("graph_consume", [], ["trajectory"], [])

        LogHelper.log_operation(pipeline_data, _Owner, load, {})
        LogHelper.log_operation(pipeline_data, _Owner, mutate, {})
        LogHelper.log_operation(pipeline_data, _Owner, consume, {})
        # Shortcut that is already implied via graph_mutate_1.
        pipeline_data.log["operations"]["graph_consume_1"]["depends_on"].append(
            "graph_load_1"
        )

        raw = LogGraphHelper.build_graph(pipeline_data.log)
        reduced = LogGraphHelper.build_graph(pipeline_data.log, reduce=True)

        assert raw.number_of_edges() == 3
        assert reduced.number_of_edges() == 2
        assert not reduced.has_edge("graph_load_1", "graph_consume_1")
        assert nx.descendants(raw, "graph_load_1") == nx.descendants(
            reduced, "graph_load_1"
        )
