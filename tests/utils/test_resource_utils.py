# mdxplain - A Python toolkit for molecular dynamics trajectory analysis
#
# Author: Maximilian Salomon
# Created with assistance from Claude Code (Claude Sonnet 4.0).
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

"""Tests for OS-compatibility behavior in ResourceUtils."""

from unittest.mock import patch

from mdxplain.utils.resource_utils import ResourceUtils


def test_apply_process_limits_io_priority_unsupported_is_safe():
    """Unsupported I/O priority should be reported without raising."""

    class DummyProc:
        def nice(self, value):
            return None

        def cpu_affinity(self, value):
            return None

    with patch(
        "mdxplain.utils.resource_utils.psutil.Process",
        return_value=DummyProc(),
    ), patch(
        "mdxplain.utils.resource_utils.ResourceUtils._apply_io_priority",
        side_effect=RuntimeError("not supported"),
    ):
        result = ResourceUtils.apply_process_limits(
            nice=0,
            io_priority="low",
            cpu_affinity=[0],
        )

    assert result["io_priority"] is None
    assert any("io_priority:" in msg for msg in result["errors"])


def test_apply_process_limits_cpu_affinity_missing_is_safe():
    """Missing cpu_affinity support should be reported without raising."""

    class DummyProc:
        def nice(self, value):
            return None

    with patch(
        "mdxplain.utils.resource_utils.psutil.Process",
        return_value=DummyProc(),
    ), patch(
        "mdxplain.utils.resource_utils.ResourceUtils._apply_io_priority",
        return_value=None,
    ):
        result = ResourceUtils.apply_process_limits(
            cpu_affinity=[0],
        )

    assert result["cpu_affinity"] is None
    assert any("cpu_affinity: not supported" in msg for msg in result["errors"])
