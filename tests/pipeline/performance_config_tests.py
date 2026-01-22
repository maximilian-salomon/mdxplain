# mdxplain - A Python toolkit for molecular dynamics trajectory analysis
#
# Author: Maximilian Salomon
# Created with assistance from Claude Code (Claude Sonnet 4.0) and GitHub Copilot (Claude Sonnet 4.0).
#
# Copyright (C) 2025 Maximilian Salomon
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

"""Tests for performance configuration and resource limit application."""

from unittest.mock import Mock, patch

import pytest

from mdxplain.pipeline.manager.performance_config import PerformanceConfig
from mdxplain.pipeline.manager.pipeline_manager import PipelineManager


def _default_perf_fields():
    return {
        "auto_resource_limits": False,
        "reserve_cores": 0,
        "resource_nice": None,
        "resource_io_priority": None,
        "resource_cpu_affinity": None,
        "auto_blas_thread_limit": False,
    }


def test_performance_config_requires_all_defaults():
    """Missing defaults should be rejected to keep config complete."""
    with pytest.raises(ValueError, match="Missing performance defaults"):
        PerformanceConfig(defaults={"auto_resource_limits": False})


def test_performance_config_update_calls_callback():
    """Changing a field should trigger a single callback call."""
    defaults = _default_perf_fields()
    on_change = Mock()
    config = PerformanceConfig(defaults=defaults, on_change=on_change)

    on_change.assert_not_called()

    config.resource_nice = 10
    on_change.assert_called_once()

    on_change.reset_mock()
    config.resource_nice = 10
    on_change.assert_not_called()

    config.update(resource_nice=11, reserve_cores=1)
    on_change.assert_called_once()


def test_performance_config_rejects_unknown_fields():
    """Unknown fields should raise to prevent silent misconfiguration."""
    defaults = _default_perf_fields()
    config = PerformanceConfig(defaults=defaults)

    with pytest.raises(AttributeError, match="Unknown performance setting"):
        config.unknown = 1

    with pytest.raises(AttributeError, match="Unknown performance setting"):
        config.update(unknown=1)


def test_performance_config_update_no_changes_no_callback():
    """Updating with identical values should not trigger callbacks."""
    defaults = _default_perf_fields()
    on_change = Mock()
    config = PerformanceConfig(defaults=defaults, on_change=on_change)

    config.update(**defaults)
    on_change.assert_not_called()


def test_pipeline_stability_default_memmap_true_applies_limits(tmp_path):
    """Default behavior enables stability when memmaps are on."""
    with patch(
        "mdxplain.pipeline.manager.pipeline_manager.ResourceUtils.recommend_cpu_affinity",
        return_value=[0, 1, 2],
        
    ) as mock_affinity, patch(
        "mdxplain.pipeline.manager.pipeline_manager.ResourceUtils.apply_process_limits",
        return_value={
            "cpu_affinity": [0, 1, 2],
            "nice": 15,
            "io_priority": "low",
            "errors": [],
        },

    ) as mock_apply, patch(
        "mdxplain.pipeline.manager.pipeline_manager.ResourceUtils.apply_blas_thread_limits",
        return_value={"max_threads": 3, "errors": []},

    ) as mock_blas:
        pipeline = PipelineManager(
            cache_dir=str(tmp_path),
            use_memmap=True,
            use_stability_config=None,
        )

    perf = pipeline.config.performance
    assert perf.auto_resource_limits is True
    assert perf.reserve_cores == 2
    assert perf.resource_nice == 15
    assert perf.resource_io_priority == "low"
    assert perf.auto_blas_thread_limit is True

    mock_affinity.assert_called_once_with(reserve_cores=2)
    mock_apply.assert_called_once_with(
        nice=15,
        io_priority="low",
        cpu_affinity=[0, 1, 2],
    )
    mock_blas.assert_called_once_with(3)


def test_pipeline_stability_default_memmap_false_keeps_os_defaults(tmp_path):
    """With memmaps disabled, default behavior should keep OS defaults."""
    with patch(
        "mdxplain.pipeline.manager.pipeline_manager.ResourceUtils.apply_process_limits"
    ) as mock_apply, patch(
        "mdxplain.pipeline.manager.pipeline_manager.ResourceUtils.apply_blas_thread_limits"
    ) as mock_blas:
        pipeline = PipelineManager(
            cache_dir=str(tmp_path),
            use_memmap=False,
            use_stability_config=None,
        )

    perf = pipeline.config.performance
    assert perf.auto_resource_limits is False
    assert perf.reserve_cores == 0
    assert perf.resource_nice is None
    assert perf.resource_io_priority is None
    assert perf.auto_blas_thread_limit is False

    mock_apply.assert_not_called()
    mock_blas.assert_not_called()


def test_pipeline_stability_disabled_even_with_memmap(tmp_path):
    """Disabling stability must keep OS defaults even with memmaps."""
    with patch(
        "mdxplain.pipeline.manager.pipeline_manager.ResourceUtils.apply_process_limits"
    ) as mock_apply, patch(
        "mdxplain.pipeline.manager.pipeline_manager.ResourceUtils.apply_blas_thread_limits"
    ) as mock_blas:
        pipeline = PipelineManager(
            cache_dir=str(tmp_path),
            use_memmap=True,
            use_stability_config=False,
        )

    perf = pipeline.config.performance
    assert perf.auto_resource_limits is False
    assert perf.reserve_cores == 0
    assert perf.resource_nice is None
    assert perf.resource_io_priority is None
    assert perf.resource_cpu_affinity is None
    assert perf.auto_blas_thread_limit is False

    mock_apply.assert_not_called()
    mock_blas.assert_not_called()


def test_pipeline_stability_explicit_true_applies_even_without_memmap(tmp_path):
    """Explicitly enabling stability applies limits regardless of memmaps."""
    with patch(
        "mdxplain.pipeline.manager.pipeline_manager.ResourceUtils.recommend_cpu_affinity",
        return_value=[0, 1],
    ) as mock_affinity, patch(
        "mdxplain.pipeline.manager.pipeline_manager.ResourceUtils.apply_process_limits",
        return_value={
            "cpu_affinity": [0, 1],
            "nice": 15,
            "io_priority": "low",
            "errors": [],
        },
    ) as mock_apply, patch(
        "mdxplain.pipeline.manager.pipeline_manager.ResourceUtils.apply_blas_thread_limits",
        return_value={"max_threads": 2, "errors": []},
    ) as mock_blas:
        pipeline = PipelineManager(
            cache_dir=str(tmp_path),
            use_memmap=False,
            use_stability_config=True,
        )

    perf = pipeline.config.performance
    assert perf.resource_io_priority == "low"

    mock_affinity.assert_called_once_with(reserve_cores=2)
    mock_apply.assert_called_once_with(
        nice=15,
        io_priority="low",
        cpu_affinity=[0, 1],
    )
    mock_blas.assert_called_once_with(2)


def test_pipeline_performance_update_triggers_apply(tmp_path):
    """Updating performance config should apply limits immediately."""
    pipeline = PipelineManager(
        cache_dir=str(tmp_path),
        use_memmap=False,
        use_stability_config=False,
    )

    with patch(
        "mdxplain.pipeline.manager.pipeline_manager.ResourceUtils.recommend_cpu_affinity",
        return_value=[0],
    ) as mock_affinity, patch(
        "mdxplain.pipeline.manager.pipeline_manager.ResourceUtils.apply_process_limits",
        return_value={
            "cpu_affinity": [0],
            "nice": 5,
            "io_priority": "normal",
            "errors": [],
        },
    ) as mock_apply, patch(
        "mdxplain.pipeline.manager.pipeline_manager.ResourceUtils.apply_blas_thread_limits",
        return_value={"max_threads": 1, "errors": []},
    ) as mock_blas:
        pipeline.config.performance.update(
            auto_resource_limits=True,
            reserve_cores=1,
            resource_nice=5,
            resource_io_priority="normal",
            auto_blas_thread_limit=True,
        )

    mock_affinity.assert_called_once_with(reserve_cores=1)
    mock_apply.assert_called_once_with(
        nice=5,
        io_priority="normal",
        cpu_affinity=[0],
    )
    mock_blas.assert_called_once_with(1)


def test_pipeline_performance_update_respects_explicit_affinity(tmp_path):
    """Explicit affinity should bypass auto selection."""
    pipeline = PipelineManager(
        cache_dir=str(tmp_path),
        use_memmap=False,
        use_stability_config=False,
    )

    with patch(
        "mdxplain.pipeline.manager.pipeline_manager.ResourceUtils.recommend_cpu_affinity"
    ) as mock_affinity, patch(
        "mdxplain.pipeline.manager.pipeline_manager.ResourceUtils.apply_process_limits",
        return_value={
            "cpu_affinity": [2, 3],
            "nice": 0,
            "io_priority": "high",
            "errors": [],
        },
    ) as mock_apply, patch(
        "mdxplain.pipeline.manager.pipeline_manager.ResourceUtils.apply_blas_thread_limits",
        return_value={"max_threads": 2, "errors": []},
    ) as mock_blas:
        pipeline.config.performance.update(
            auto_resource_limits=False,
            reserve_cores=0,
            resource_nice=0,
            resource_io_priority="high",
            resource_cpu_affinity=[2, 3],
            auto_blas_thread_limit=True,
        )

    mock_affinity.assert_not_called()
    mock_apply.assert_called_once_with(
        nice=0,
        io_priority="high",
        cpu_affinity=[2, 3],
    )
    mock_blas.assert_called_once_with(2)


def test_pipeline_performance_update_with_auto_and_explicit_affinity(tmp_path):
    """Explicit affinity should be used even if auto_resource_limits is True."""
    pipeline = PipelineManager(
        cache_dir=str(tmp_path),
        use_memmap=False,
        use_stability_config=False,
    )

    with patch(
        "mdxplain.pipeline.manager.pipeline_manager.ResourceUtils.recommend_cpu_affinity"
    ) as mock_affinity, patch(
        "mdxplain.pipeline.manager.pipeline_manager.ResourceUtils.apply_process_limits",
        return_value={
            "cpu_affinity": [4, 5, 6],
            "nice": 3,
            "io_priority": "low",
            "errors": [],
        },
    ) as mock_apply, patch(
        "mdxplain.pipeline.manager.pipeline_manager.ResourceUtils.apply_blas_thread_limits",
        return_value={"max_threads": 3, "errors": []},
    ) as mock_blas:
        pipeline.config.performance.update(
            auto_resource_limits=True,
            reserve_cores=1,
            resource_nice=3,
            resource_io_priority="low",
            resource_cpu_affinity=[4, 5, 6],
            auto_blas_thread_limit=True,
        )

    mock_affinity.assert_not_called()
    mock_apply.assert_called_once_with(
        nice=3,
        io_priority="low",
        cpu_affinity=[4, 5, 6],
    )
    mock_blas.assert_called_once_with(3)


def test_pipeline_performance_update_io_priority_only(tmp_path):
    """Setting only I/O priority should apply limits without affinity."""
    pipeline = PipelineManager(
        cache_dir=str(tmp_path),
        use_memmap=False,
        use_stability_config=False,
    )

    with patch(
        "mdxplain.pipeline.manager.pipeline_manager.ResourceUtils.apply_process_limits",
        return_value={
            "cpu_affinity": None,
            "nice": None,
            "io_priority": "idle",
            "errors": [],
        },
    ) as mock_apply, patch(
        "mdxplain.pipeline.manager.pipeline_manager.ResourceUtils.apply_blas_thread_limits"
    ) as mock_blas:
        pipeline.config.performance.update(
            resource_io_priority="idle",
        )

    mock_apply.assert_called_once_with(
        nice=None,
        io_priority="idle",
        cpu_affinity=None,
    )
    mock_blas.assert_called_once_with(None)


def test_pipeline_performance_update_disables_blas_limits(tmp_path):
    """Disabling BLAS thread limits should reset them to defaults."""
    pipeline = PipelineManager(
        cache_dir=str(tmp_path),
        use_memmap=False,
        use_stability_config=False,
    )

    with patch(
        "mdxplain.pipeline.manager.pipeline_manager.ResourceUtils.apply_process_limits"
    ) as mock_apply, patch(
        "mdxplain.pipeline.manager.pipeline_manager.ResourceUtils.apply_blas_thread_limits",
        return_value={"max_threads": None, "errors": []},
    ) as mock_blas:
        pipeline.config.performance.update(
            resource_nice=1,
            auto_blas_thread_limit=False,
        )

    mock_apply.assert_called_once_with(
        nice=1,
        io_priority=None,
        cpu_affinity=None,
    )
    mock_blas.assert_called_once_with(None)


def test_pipeline_performance_update_blas_without_affinity(tmp_path):
    """BLAS limits should fall back to allowed CPUs when affinity is unknown."""
    pipeline = PipelineManager(
        cache_dir=str(tmp_path),
        use_memmap=False,
        use_stability_config=False,
    )

    with patch(
        "mdxplain.pipeline.manager.pipeline_manager.ResourceUtils.recommend_cpu_affinity",
        return_value=None,
    ) as mock_affinity, patch(
        "mdxplain.pipeline.manager.pipeline_manager.ResourceUtils.apply_blas_thread_limits"
    ) as mock_blas:
        pipeline.config.performance.update(
            auto_resource_limits=False,
            reserve_cores=0,
            resource_nice=None,
            resource_io_priority=None,
            auto_blas_thread_limit=True,
        )

    mock_affinity.assert_called_once_with(reserve_cores=0)
    mock_blas.assert_not_called()
