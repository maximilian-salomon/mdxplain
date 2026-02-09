# mdxplain - A Python toolkit for molecular dynamics trajectory analysis
#
# Author: Maximilian Salomon
# Created with assistance from Codex GPT-5.
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

"""Integration tests for parallel pipeline isolation with shared base cache."""

from concurrent.futures import ThreadPoolExecutor
import multiprocessing as mp
import os
from pathlib import Path
from typing import Dict, Any

import numpy as np

from mdxplain.feature.feature_type.distances import Distances
from mdxplain.pipeline.manager.pipeline_manager import PipelineManager
from tests.fixtures.mock_trajectory_factory import MockTrajectoryFactory


def _assign_mock_trajectory(pipeline: PipelineManager, seed: int) -> None:
    """Attach a small mock trajectory to the pipeline for feature tests."""
    # Keep trajectory small so tests remain fast and deterministic.
    # The test goal is cache/memmap isolation, not numerical heavy workload.
    mock = MockTrajectoryFactory.create_simple(n_frames=64, n_atoms=10, seed=seed)
    # Inject trajectory directly so this test does not depend on I/O loaders.
    pipeline._data.trajectory_data.trajectories = [mock]
    # Stable, unique name per seed makes debugging failures easier.
    pipeline._data.trajectory_data.trajectory_names = [f"traj_{seed}"]
    # Mirror the key metadata fields expected by feature calculators.
    pipeline._data.trajectory_data.n_frames = mock.n_frames
    pipeline._data.trajectory_data.n_atoms = mock.n_atoms
    # Distances feature needs residue labels; build a minimal valid mapping.
    pipeline._data.trajectory_data.res_label_data = {
        0: [{"seqid": i, "full_name": f"RES_{i}"} for i in range(mock.n_atoms)]
    }


def _compute_distances_worker(base_cache: str, seed: int, queue: mp.Queue) -> None:
    """Process worker used for multiprocessing isolation test."""
    # Initialize for robust finally-close even if construction fails.
    pipeline = None
    try:
        # Each process builds its own manager instance in the same base cache root.
        # Correct behavior: manager derives an isolated scoped cache path.
        pipeline = PipelineManager(use_memmap=True, cache_dir=base_cache, show_progress=False)
        # Attach per-worker test trajectory.
        _assign_mock_trajectory(pipeline, seed=seed)
        # Trigger memmap-backed feature creation.
        pipeline.feature.add_feature(Distances(), force=True)
        feature_data = pipeline._data.feature_data["distances"][0]
        # Return only serializable diagnostics for parent assertions.
        queue.put(
            {
                # "ok" allows parent to distinguish worker exception from success.
                "ok": True,
                # Runtime scoped cache directory selected by this pipeline instance.
                "cache_dir": pipeline.get_config()["cache_dir"],
                # Concrete cache file path where memmap data is stored.
                "cache_path": feature_data.cache_path,
                # Shape confirms both workers computed equivalent data dimensions.
                "shape": tuple(feature_data.data.shape),
            }
        )
    except Exception as exc:  # pragma: no cover - only exercised on failure path
        # Surface worker exceptions to parent process for readable failure diagnostics.
        queue.put({"ok": False, "error": repr(exc)})
    finally:
        if pipeline is not None:
            # Important on Windows: release memmap handles before process exit.
            pipeline.close()


class TestPipelineParallelIsolation:
    """Validate parallel isolation across pipelines using a shared base cache dir."""

    def _build_pipeline(self, base_cache: Path, seed: int) -> PipelineManager:
        """Create a memmap pipeline with one mock trajectory."""
        # Create a fresh pipeline that points at shared base cache.
        # Correct behavior: each pipeline gets a unique scoped subdirectory.
        pipeline = PipelineManager(
            use_memmap=True,
            cache_dir=str(base_cache),
            show_progress=False,
        )
        # Populate minimal trajectory state needed for feature computation.
        _assign_mock_trajectory(pipeline, seed=seed)
        return pipeline

    def test_parallel_threads_same_base_cache_create_isolated_outputs(self, tmp_path):
        """Two pipelines in parallel threads should write to different scoped cache dirs."""
        # One shared base dir simulates real-world concurrent usage.
        base_cache = tmp_path / "shared_cache"
        # Two independent pipelines intentionally target the same base cache.
        p1 = self._build_pipeline(base_cache, seed=100)
        p2 = self._build_pipeline(base_cache, seed=200)

        try:
            def _run_compute(pipeline: PipelineManager) -> Dict[str, Any]:
                # Compute one memmap-backed feature and return cache metadata.
                pipeline.feature.add_feature(Distances(), force=True)
                feature = pipeline._data.feature_data["distances"][0]
                return {
                    # Where this pipeline currently writes cache artifacts.
                    "cache_dir": pipeline.get_config()["cache_dir"],
                    # Concrete memmap file for the distances feature.
                    "cache_path": feature.cache_path,
                    # Light numeric check proving real data processing happened.
                    "mean": float(np.mean(feature.data)),
                }

            with ThreadPoolExecutor(max_workers=2) as executor:
                # Run both pipelines concurrently against the same base cache root.
                f1 = executor.submit(_run_compute, p1)
                f2 = executor.submit(_run_compute, p2)
                r1 = f1.result(timeout=120)
                r2 = f2.result(timeout=120)

            # The scoped runtime dirs must differ even when base dir is shared.
            assert r1["cache_dir"] != r2["cache_dir"]
            # Each feature file must live inside its own pipeline-scoped cache dir.
            assert os.path.dirname(r1["cache_path"]) == r1["cache_dir"]
            assert os.path.dirname(r2["cache_path"]) == r2["cache_dir"]
            # Both scoped dirs should still be children of the same base cache root.
            assert os.path.dirname(r1["cache_dir"]) == os.path.abspath(os.path.normpath(base_cache))
            assert os.path.dirname(r2["cache_dir"]) == os.path.abspath(os.path.normpath(base_cache))
            # Sanity-check that real numeric outputs were produced.
            assert np.isfinite(r1["mean"])
            assert np.isfinite(r2["mean"])
        finally:
            # Cleanup in finally keeps tests stable even on assertion failures.
            p1.close()
            p2.close()

    def test_close_one_pipeline_during_other_compute_keeps_other_usable(self, tmp_path):
        """Closing one pipeline must not break a concurrent compute in another pipeline."""
        base_cache = tmp_path / "shared_cache"
        # Build two independent pipelines sharing only the base cache root.
        p1 = self._build_pipeline(base_cache, seed=300)
        p2 = self._build_pipeline(base_cache, seed=400)

        try:
            # Ensure p1 has active memmap-backed data before close().
            p1.feature.add_feature(Distances(), force=True)

            def _run_p2() -> int:
                # This compute runs while p1 is being closed.
                p2.feature.add_feature(Distances(), force=True)
                # Return feature width as a simple "still usable" signal.
                return p2._data.feature_data["distances"][0].analysis.compute_mean().shape[0]

            with ThreadPoolExecutor(max_workers=1) as executor:
                # Start p2 compute asynchronously.
                future = executor.submit(_run_p2)
                # Closing p1 must not close memmaps owned by p2.
                p1.close()
                # Timeout avoids hanging test in case of deadlock/lock contention.
                p2_width = future.result(timeout=120)

            # Non-zero width confirms successful compute and usable output.
            assert p2_width > 0
        finally:
            # Idempotent close is intentional; safe even if p1 already closed above.
            p1.close()
            p2.close()

    def test_parallel_processes_same_base_cache_remain_isolated(self, tmp_path):
        """Two separate processes should compute independently with shared base cache root."""
        # Separate-process scenario catches issues hidden in single-process tests.
        base_cache = tmp_path / "shared_cache_proc"
        # "spawn" mirrors Windows process semantics and avoids inherited state.
        ctx = mp.get_context("spawn")
        # Parent receives small diagnostic payloads from workers.
        queue: mp.Queue = ctx.Queue()

        # Launch worker process 1.
        proc1 = ctx.Process(
            target=_compute_distances_worker,
            args=(str(base_cache), 501, queue),
        )
        # Launch worker process 2.
        proc2 = ctx.Process(
            target=_compute_distances_worker,
            args=(str(base_cache), 502, queue),
        )
        # Start both workers.
        proc1.start()
        proc2.start()
        # Bound runtime to fail fast on deadlocks/endless loops.
        proc1.join(timeout=180)
        proc2.join(timeout=180)

        # Workers should exit cleanly.
        assert proc1.exitcode == 0
        assert proc2.exitcode == 0

        # Retrieve per-worker diagnostics and verify cache isolation.
        result1 = queue.get(timeout=30)
        result2 = queue.get(timeout=30)
        # Both workers should report success.
        assert result1["ok"] is True
        assert result2["ok"] is True
        # Core isolation check: each process got its own scoped cache dir.
        assert result1["cache_dir"] != result2["cache_dir"]
        # Memmap files must live in their corresponding scoped directories.
        assert os.path.dirname(result1["cache_path"]) == result1["cache_dir"]
        assert os.path.dirname(result2["cache_path"]) == result2["cache_dir"]
        # Both workers should compute compatible output dimensions.
        assert tuple(result1["shape"]) == tuple(result2["shape"])
