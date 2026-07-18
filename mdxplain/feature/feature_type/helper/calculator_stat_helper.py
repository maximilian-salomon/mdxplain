# mdxplain - A Python toolkit for molecular dynamics trajectory analysis
#
# Author: Maximilian Salomon
# Created with assistance from Cursor IDE (Claude Sonnet 4.0, occasional Claude Sonnet 3.7 and Gemini 2.5 Pro).
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

"""
Statistical calculations for molecular dynamics feature data.

Provides statistical calculations for feature data with support for memory-mapped
arrays and chunked processing. All methods are static and can be used without
instantiation across different calculators.
"""

from typing import Any, Callable, Generator, Iterator, Optional

import numpy as np

from mdxplain.utils.progress_utils import ProgressUtils
from mdxplain.utils.resource_utils import ResourceUtils
from mdxplain.utils.memmap_utils import MemmapUtils

from .feature_shape_helper import FeatureShapeHelper


class CalculatorStatHelper:
    """
    Static utility class for statistical calculations on molecular dynamics feature data.

    Provides efficient statistical computations (mean, std, transitions, etc.) with
    support for memory-mapped arrays and chunked processing for large datasets.
    All methods are static for easy use across different calculators.

    Attributes
    ----------
    STREAMING_REDUCTIONS : frozenset
        Reductions that can be accumulated across frame blocks. Everything else
        (median, percentiles, MAD) needs every frame of a column at once and must
        chunk the feature axis instead. Derived from the accumulator registry, so
        it cannot drift out of step with what is actually implemented.
    """

    # Maps each streamable reduction to the accumulator family that owns its
    # algebra. Single source of truth: STREAMING_REDUCTIONS is derived from it, so
    # a reduction can never be accepted without an accumulator registered for it.
    _STREAM_FAMILIES = {
        "sum": "sum",
        "mean": "sum",
        "min": "extremes",
        "max": "extremes",
        "ptp": "extremes",
        "var": "moment",
        "std": "moment",
    }

    STREAMING_REDUCTIONS = frozenset(_STREAM_FAMILIES)

    # Each per-residue metric names the per-column reductions it needs and how to
    # fold a residue's columns into one value. Equal frame counts per column make
    # the sums, extrema and law-of-total-variance folds exact. 'median' is absent:
    # it cannot be composed from column summaries and is handled separately.
    _PER_RESIDUE_FOLDS = {
        "sum": (("sum",), lambda cols, stats: float(stats["sum"][cols].sum())),
        "mean": (("mean",), lambda cols, stats: float(stats["mean"][cols].mean())),
        "min": (("min",), lambda cols, stats: float(stats["min"][cols].min())),
        "max": (("max",), lambda cols, stats: float(stats["max"][cols].max())),
        "range": (
            ("min", "max"),
            lambda cols, stats: float(
                stats["max"][cols].max() - stats["min"][cols].min()
            ),
        ),
        "variance": (
            ("mean", "var"),
            lambda cols, stats: CalculatorStatHelper._residue_variance(cols, stats),
        ),
        "std": (
            ("mean", "var"),
            lambda cols, stats: float(
                np.sqrt(CalculatorStatHelper._residue_variance(cols, stats))
            ),
        ),
    }

    @staticmethod
    def _residue_variance(cols: np.ndarray, stats: dict) -> float:
        """
        Pool per-column mean and variance into a residue variance.

        Every column spans the same frames, so the pooled population variance is
        the mean of the column variances plus the variance of the column means —
        the law of total variance for equal group sizes.

        Parameters
        ----------
        cols : numpy.ndarray
            Column indices of the residue
        stats : dict
            Per-column 'mean' and 'var' arrays

        Returns
        -------
        float
            Pooled variance over the residue's columns and frames
        """
        return float(stats["var"][cols].mean() + stats["mean"][cols].var())

    # ===== CHUNK SIZING =====

    @staticmethod
    def resolve_output_block_size(
        chunk_size: Optional[int], n_rows: int, n_units: int
    ) -> int:
        """
        Convert a frame-based chunk size into an equivalent output-block width.

        ``chunk_size`` counts frames. A reduction that has to keep every frame of
        a column in memory (median, MAD, transitions) cannot chunk frames, so it
        shrinks its output axis instead. The budget is the memory one full frame
        chunk would occupy — ``chunk_size`` frames across all units — and dividing
        it by the cost of a single unit across all rows gives the number of units
        that fit in the same budget. The itemsize cancels on both sides and
        therefore does not appear.

        Mirrors the frames-to-atoms conversion in
        ``ParallelOperationsHelper._calculate_atom_chunk_size``.

        Parameters
        ----------
        chunk_size : int, optional
            Number of frames per chunk; falsy means a single block over all units
        n_rows : int
            Total number of frames the reduction spans
        n_units : int
            Number of units on the output axis (features, residues, ...)

        Returns
        -------
        int
            Block width in units, at least 1 and at most n_units

        Examples
        --------
        >>> CalculatorStatHelper.resolve_output_block_size(2000, 1000000, 50000)
        100
        """
        if n_units <= 0:
            return 1
        if not chunk_size or n_rows <= 0:
            return n_units
        block = (chunk_size * n_units) // n_rows
        return max(1, min(n_units, block))

    # ===== STREAMING PER-FEATURE REDUCTIONS =====

    @staticmethod
    def compute_reduction_per_feature(
        array: np.ndarray,
        reduction: str,
        chunk_size: Optional[int] = 2000,
        use_memmap: bool = False,
        transform: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    ) -> np.ndarray:
        """
        Stream a per-feature reduction over frame blocks.

        Reductions that can be accumulated are computed by walking the frame axis
        and combining partial results, so peak memory is ``chunk_size`` frames by
        the full feature width — the budget ``chunk_size`` actually describes.

        There is deliberately no separate "unchunked" formula: a small or in-RAM
        array simply yields a single block. Chunking therefore cannot change the
        result, only the block size.

        Parameters
        ----------
        array : numpy.ndarray
            Feature array with shape (n_frames, n_features) or (n_frames, M, M)
        reduction : str
            One of 'sum', 'mean', 'min', 'max', 'ptp', 'var', 'std'
        chunk_size : int, optional
            Number of frames per block; falsy means a single block
        use_memmap : bool, default=False
            Force block-wise processing even for in-RAM arrays
        transform : callable, optional
            Applied to each block before reduction. Use this instead of
            transforming the whole array up front, which would defeat chunking.

        Returns
        -------
        numpy.ndarray
            Reduction value per feature, preserving the spatial shape

        Raises
        ------
        ValueError
            If reduction is not a streamable reduction

        Examples
        --------
        >>> array = np.array([[1.0, 2.0], [3.0, 4.0]])
        >>> CalculatorStatHelper.compute_reduction_per_feature(array, "mean")
        array([2., 3.])
        """
        if reduction not in CalculatorStatHelper.STREAMING_REDUCTIONS:
            raise ValueError(
                f"Unknown reduction: {reduction}. "
                f"Supported: {sorted(CalculatorStatHelper.STREAMING_REDUCTIONS)}"
            )
        spatial_shape = array.shape[1:]
        flat = array.reshape(array.shape[0], -1) if array.ndim > 2 else array
        blocks = CalculatorStatHelper._iter_frame_blocks(
            flat, chunk_size, use_memmap, transform
        )
        result = CalculatorStatHelper._reduce_blocks(blocks, reduction)
        if result.size == 0:
            return result
        return result.reshape(spatial_shape)

    @staticmethod
    def _reduce_blocks(blocks: Iterator[np.ndarray], reduction: str) -> np.ndarray:
        """
        Dispatch a streamed reduction to its accumulator.

        Parameters
        ----------
        blocks : iterator
            Iterator over (n_block_frames, n_features) float64 blocks
        reduction : str
            Streamable reduction name

        Returns
        -------
        numpy.ndarray
            Reduction value per feature

        Raises
        ------
        ValueError
            If the reduction's family has no accumulator
        """
        family = CalculatorStatHelper._STREAM_FAMILIES[reduction]
        if family == "sum":
            return CalculatorStatHelper._stream_sum(blocks, reduction)
        if family == "extremes":
            return CalculatorStatHelper._stream_extremes(blocks, reduction)
        if family == "moment":
            return CalculatorStatHelper._stream_moment(blocks, reduction)
        raise ValueError(f"No accumulator registered for family '{family}'.")

    @staticmethod
    def _resolve_frame_step(
        array: np.ndarray, chunk_size: Optional[int], use_memmap: bool
    ) -> int:
        """
        Decide how many frames one block holds.

        Parameters
        ----------
        array : numpy.ndarray
            Feature array being reduced
        chunk_size : int, optional
            Requested frames per block
        use_memmap : bool
            Force block-wise processing

        Returns
        -------
        int
            Frames per block, at least 1
        """
        n_rows = array.shape[0]
        should_chunk = use_memmap or FeatureShapeHelper.is_memmap(array)
        if not should_chunk or not chunk_size:
            return max(1, n_rows)
        return max(1, min(chunk_size, n_rows))

    @staticmethod
    def _iter_frame_blocks(
        array: np.ndarray,
        chunk_size: Optional[int],
        use_memmap: bool,
        transform: Optional[Callable[[np.ndarray], np.ndarray]],
    ) -> Generator[np.ndarray, None, None]:
        """
        Yield float64 frame blocks, applying the transform inside the loop.

        Parameters
        ----------
        array : numpy.ndarray
            Flattened feature array with shape (n_frames, n_features)
        chunk_size : int, optional
            Requested frames per block
        use_memmap : bool
            Force block-wise processing
        transform : callable, optional
            Applied to each block before it is yielded

        Yields
        ------
        numpy.ndarray
            One float64 block of frames
        """
        n_rows = array.shape[0]
        step = CalculatorStatHelper._resolve_frame_step(array, chunk_size, use_memmap)
        is_memmap_input = MemmapUtils.is_memmap_view(array)
        if is_memmap_input:
            ResourceUtils.tune_memmap(array, "sequential")
        try:
            for start in ProgressUtils.iterate(
                range(0, n_rows, step),
                desc="Computing statistics per feature",
                unit="chunks",
            ):
                block = np.asarray(
                    array[start : min(start + step, n_rows)], dtype=np.float64
                )
                yield transform(block) if transform is not None else block
        finally:
            # Also runs when a consumer abandons the generator early, which would
            # otherwise leave the sequential-access hint stuck on the memmap.
            if is_memmap_input:
                ResourceUtils.tune_memmap(array, "random")

    @staticmethod
    def _stream_sum(blocks: Iterator[np.ndarray], reduction: str) -> np.ndarray:
        """
        Accumulate a sum, optionally normalising it to a mean.

        Parameters
        ----------
        blocks : iterator
            Iterator over float64 frame blocks
        reduction : str
            Either 'sum' or 'mean'

        Returns
        -------
        numpy.ndarray
            Sum or mean per feature
        """
        total, count = CalculatorStatHelper._accumulate_sum(blocks)
        if total is None:
            return np.array([])
        if reduction == "sum":
            return total
        return total / count

    @staticmethod
    def _accumulate_sum(blocks: Iterator[np.ndarray]) -> tuple:
        """
        Add up frame blocks and count the frames seen.

        Parameters
        ----------
        blocks : iterator
            Iterator over float64 frame blocks

        Returns
        -------
        tuple
            (total per feature or None if there were no blocks, frame count)
        """
        total = None
        count = 0
        for block in blocks:
            block_sum = block.sum(axis=0)
            total = block_sum if total is None else total + block_sum
            count += block.shape[0]
        return total, count

    @staticmethod
    def _stream_extremes(blocks: Iterator[np.ndarray], reduction: str) -> np.ndarray:
        """
        Accumulate running minima and maxima.

        Parameters
        ----------
        blocks : iterator
            Iterator over float64 frame blocks
        reduction : str
            One of 'min', 'max', 'ptp'

        Returns
        -------
        numpy.ndarray
            Minimum, maximum or peak-to-peak range per feature
        """
        lowest, highest = CalculatorStatHelper._accumulate_extremes(blocks)
        if lowest is None:
            return np.array([])
        if reduction == "min":
            return lowest
        if reduction == "max":
            return highest
        return highest - lowest

    @staticmethod
    def _accumulate_extremes(blocks: Iterator[np.ndarray]) -> tuple:
        """
        Track running minima and maxima across frame blocks.

        Parameters
        ----------
        blocks : iterator
            Iterator over float64 frame blocks

        Returns
        -------
        tuple
            (minimum per feature, maximum per feature); both None if there were
            no blocks
        """
        lowest = None
        highest = None
        for block in blocks:
            block_low = block.min(axis=0)
            block_high = block.max(axis=0)
            lowest = block_low if lowest is None else np.minimum(lowest, block_low)
            highest = block_high if highest is None else np.maximum(highest, block_high)
        return lowest, highest

    @staticmethod
    def _stream_moment(blocks: Iterator[np.ndarray], reduction: str) -> np.ndarray:
        """
        Accumulate variance with Chan's numerically stable parallel combination.

        Deliberately not ``E[x^2] - E[x]^2``: molecular feature columns routinely
        have a mean far larger than their spread (a distance of 5 nm varying by
        0.01 nm), where that one-pass form loses most of its significant digits.

        Parameters
        ----------
        blocks : iterator
            Iterator over float64 frame blocks
        reduction : str
            Either 'var' or 'std'

        Returns
        -------
        numpy.ndarray
            Population variance (ddof=0) or its square root per feature
        """
        count = 0
        mean = None
        sum_squares = None
        for block in blocks:
            count, mean, sum_squares = CalculatorStatHelper._merge_moment(
                count, mean, sum_squares, block
            )
        if mean is None or count == 0:
            return np.array([])
        variance = sum_squares / count
        return variance if reduction == "var" else np.sqrt(variance)

    @staticmethod
    def _merge_moment(
        count: int,
        mean: Optional[np.ndarray],
        sum_squares: Optional[np.ndarray],
        block: np.ndarray,
    ) -> tuple:
        """
        Merge one block into a running (count, mean, M2) moment state.

        Parameters
        ----------
        count : int
            Frames accumulated so far
        mean : numpy.ndarray, optional
            Running mean per feature, None before the first block
        sum_squares : numpy.ndarray, optional
            Running M2 (sum of squared deviations) per feature
        block : numpy.ndarray
            Next float64 frame block

        Returns
        -------
        tuple
            Updated (count, mean, sum_squares)
        """
        block_count = block.shape[0]
        block_mean = block.mean(axis=0)
        block_squares = ((block - block_mean) ** 2).sum(axis=0)
        if mean is None:
            return block_count, block_mean, block_squares
        total = count + block_count
        delta = block_mean - mean
        merged_squares = (
            sum_squares + block_squares + delta**2 * count * block_count / total
        )
        return total, mean + delta * block_count / total, merged_squares

    # ===== POOLED PER-FEATURE REDUCTIONS =====

    @staticmethod
    def compute_pooled_reduction_per_feature(
        segments: list,
        reduction: str,
        chunk_size: Optional[int] = 2000,
        use_memmap: bool = False,
        transform: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    ) -> np.ndarray:
        """
        Stream a per-feature reduction across the frames of every segment.

        Pooling segments along the frame axis is exactly "more frames" for a
        per-feature reduction, so the accumulator does not care which segment a
        block came from. Nothing is concatenated: peak memory stays at
        ``chunk_size`` frames by the feature width, no matter how many segments
        there are or how long they are.

        Parameters
        ----------
        segments : list
            List of (n_frames, n_features) arrays to pool along the frame axis
        reduction : str
            One of 'sum', 'mean', 'min', 'max', 'ptp', 'var', 'std'
        chunk_size : int, optional
            Number of frames per block; falsy means one block per segment
        use_memmap : bool, default=False
            Force block-wise processing even for in-RAM arrays
        transform : callable, optional
            Applied to each block before reduction

        Returns
        -------
        numpy.ndarray
            Reduction value per feature, empty if there are no segments

        Examples
        --------
        >>> segments = [np.array([[1.0, 2.0]]), np.array([[3.0, 4.0]])]
        >>> CalculatorStatHelper.compute_pooled_reduction_per_feature(segments, "mean")
        array([2., 3.])
        """
        if reduction not in CalculatorStatHelper.STREAMING_REDUCTIONS:
            raise ValueError(
                f"Unknown reduction: {reduction}. "
                f"Supported: {sorted(CalculatorStatHelper.STREAMING_REDUCTIONS)}"
            )
        if not segments:
            return np.array([])
        spatial_shape = segments[0].shape[1:]
        blocks = CalculatorStatHelper._iter_pooled_frame_blocks(
            segments, chunk_size, use_memmap, transform
        )
        result = CalculatorStatHelper._reduce_blocks(blocks, reduction)
        if result.size == 0:
            return result
        return result.reshape(spatial_shape)

    @staticmethod
    def _iter_pooled_frame_blocks(
        segments: list,
        chunk_size: Optional[int],
        use_memmap: bool,
        transform: Optional[Callable[[np.ndarray], np.ndarray]],
    ) -> Generator[np.ndarray, None, None]:
        """
        Chain the frame blocks of every segment into one stream.

        Parameters
        ----------
        segments : list
            List of (n_frames, n_features) arrays
        chunk_size : int, optional
            Requested frames per block
        use_memmap : bool
            Force block-wise processing
        transform : callable, optional
            Applied to each block before it is yielded

        Yields
        ------
        numpy.ndarray
            One float64 block of frames, from one segment at a time
        """
        for segment in segments:
            flat = (
                segment.reshape(segment.shape[0], -1)
                if segment.ndim > 2
                else segment
            )
            yield from CalculatorStatHelper._iter_frame_blocks(
                flat, chunk_size, use_memmap, transform
            )

    @staticmethod
    def compute_pooled_func_per_feature(
        segments: list,
        func: Callable[[np.ndarray], np.ndarray],
        chunk_size: Optional[int] = 2000,
        use_memmap: bool = False,
    ) -> np.ndarray:
        """
        Apply a per-feature reduction to segments pooled one feature block at a time.

        For reductions that need every frame of a column at once (median, MAD) the
        frames cannot be streamed, so only a block of columns is pooled and the
        full (total_frames, n_features) array never exists.

        Every value must depend only on its own column. Metrics whose features come
        in fixed groups (the x/y/z triplet of one atom) would be split by an
        arbitrary block boundary and must not use this path.

        Parameters
        ----------
        segments : list
            List of (n_frames, n_features) arrays to pool along the frame axis
        func : callable
            Maps a pooled (total_frames, block_width) array to one value per feature
        chunk_size : int, optional
            Number of frames per chunk; converted to a feature-block width
        use_memmap : bool, default=False
            Force block-wise processing

        Returns
        -------
        numpy.ndarray
            Reduction value per feature, empty if there are no segments

        Examples
        --------
        >>> segments = [np.array([[1.0, 2.0]]), np.array([[3.0, 4.0]])]
        >>> CalculatorStatHelper.compute_pooled_func_per_feature(
        ...     segments, lambda block: np.median(block, axis=0)
        ... )
        array([2., 3.])
        """
        if not segments:
            return np.array([])
        n_features = segments[0].shape[1]
        if n_features == 0:
            return np.array([])
        block = CalculatorStatHelper._resolve_pooled_block_size(
            segments, chunk_size, use_memmap
        )
        result_chunks = []
        for start in ProgressUtils.iterate(
            range(0, n_features, block),
            desc="Computing pooled statistics per feature",
            unit="chunks",
        ):
            end = min(start + block, n_features)
            pooled_block = np.concatenate(
                [segment[:, start:end] for segment in segments], axis=0
            )
            result_chunks.append(np.asarray(func(pooled_block)))
        return np.concatenate(result_chunks)

    @staticmethod
    def _resolve_pooled_block_size(
        segments: list, chunk_size: Optional[int], use_memmap: bool
    ) -> int:
        """
        Decide how many features one pooled block holds.

        The row count is the pooled total across all segments, since that is how
        many frames a single column costs once the segments are stacked.

        Parameters
        ----------
        segments : list
            List of (n_frames, n_features) arrays
        chunk_size : int, optional
            Requested frames per chunk
        use_memmap : bool
            Force block-wise processing

        Returns
        -------
        int
            Features per block, at least 1 and at most n_features
        """
        n_features = segments[0].shape[1]
        should_chunk = use_memmap or any(
            FeatureShapeHelper.is_memmap(segment) for segment in segments
        )
        if not should_chunk:
            return n_features
        total_rows = sum(int(segment.shape[0]) for segment in segments)
        return CalculatorStatHelper.resolve_output_block_size(
            chunk_size, total_rows, n_features
        )

    # ===== PER-RESIDUE REDUCTIONS =====

    @staticmethod
    def compute_per_residue_reduction(
        data: np.ndarray,
        pairs: Optional[list],
        n_residues: Optional[int],
        metric: str,
        chunk_size: Optional[int] = 2000,
        use_memmap: bool = False,
    ) -> np.ndarray:
        """
        Reduce condensed pair data to one value per residue.

        Each residue is reduced over the columns it participates in — its real
        partners from ``pairs`` — pooling every frame of those columns. The
        squareform is never built and the self-distance diagonal never enters,
        so ``min`` is a residue's closest real contact rather than a constant
        zero, and ``mean`` is not diluted by that zero.

        Streamable metrics are composed from per-column statistics, which stream
        over frames, so peak memory is one frame block by the pair width. Only
        ``median`` needs a residue's columns in full and gathers them per residue.

        Parameters
        ----------
        data : numpy.ndarray
            Condensed pair array with shape (n_frames, n_pairs)
        pairs : list, optional
            Residue index pair (a, b) for each condensed column, in column order.
            None assumes a full upper triangle inferred from the column count.
        n_residues : int, optional
            Number of residues; the length of the returned array. Inferred from
            pairs when omitted.
        metric : str
            One of 'mean', 'std', 'variance', 'min', 'max', 'sum', 'range',
            'median'
        chunk_size : int, optional
            Number of frames per block
        use_memmap : bool, default=False
            Force block-wise processing

        Returns
        -------
        numpy.ndarray
            Metric value per residue; residues with no retained partner are 0

        Raises
        ------
        ValueError
            If metric is not a supported per-residue metric
        """
        pairs, n_residues = CalculatorStatHelper._resolve_pairs(
            pairs, n_residues, data.shape[1]
        )
        columns_by_residue = CalculatorStatHelper._residue_column_map(pairs, n_residues)
        if metric == "median":
            return CalculatorStatHelper._per_residue_median(data, columns_by_residue)
        return CalculatorStatHelper._per_residue_from_columns(
            data, columns_by_residue, metric, chunk_size, use_memmap
        )

    @staticmethod
    def _resolve_pairs(
        pairs: Optional[list], n_residues: Optional[int], n_pairs: int
    ) -> tuple:
        """
        Fill in the residue pairs and count when a caller omits them.

        A caller holding the real pairs (the analysis service) passes them through
        unchanged. A caller with only the condensed array supplies neither, and the
        one inference possible from the column count is a full upper triangle.

        Parameters
        ----------
        pairs : list, optional
            Residue index pairs in column order, or None to infer a full triangle
        n_residues : int, optional
            Residue count, or None to derive it from the pairs
        n_pairs : int
            Number of condensed columns, used when inferring a full triangle

        Returns
        -------
        tuple
            (pairs, n_residues), both filled in
        """
        if pairs is None:
            return CalculatorStatHelper._full_triangle_pairs(n_pairs)
        if n_residues is None:
            n_residues = max((max(pair) for pair in pairs), default=-1) + 1
        return pairs, n_residues

    @staticmethod
    def _full_triangle_pairs(n_pairs: int) -> tuple:
        """
        Reconstruct the pair list of a full upper triangle from its column count.

        Only valid when no residue pair was excluded; the fallback when a caller
        supplies no explicit pair list.

        Parameters
        ----------
        n_pairs : int
            Number of condensed columns

        Returns
        -------
        tuple
            The (i, j) pairs in column order and the residue count
        """
        n_residues = int((1 + np.sqrt(1 + 8 * n_pairs)) / 2)
        pairs = [
            (i, j)
            for i in range(n_residues)
            for j in range(i + 1, n_residues)
        ]
        return pairs, n_residues

    @staticmethod
    def _residue_column_map(pairs: list, n_residues: int) -> list:
        """
        List the condensed columns each residue participates in.

        Parameters
        ----------
        pairs : list
            Residue index pair (a, b) for each condensed column, in column order
        n_residues : int
            Number of residues

        Returns
        -------
        list
            columns_by_residue[r] is an int array of the columns touching residue r
        """
        columns_by_residue: list = [[] for _ in range(n_residues)]
        for col_idx, (residue_a, residue_b) in enumerate(pairs):
            columns_by_residue[residue_a].append(col_idx)
            columns_by_residue[residue_b].append(col_idx)
        return [np.asarray(cols, dtype=int) for cols in columns_by_residue]

    @staticmethod
    def _per_residue_from_columns(
        data: np.ndarray,
        columns_by_residue: list,
        metric: str,
        chunk_size: Optional[int],
        use_memmap: bool,
    ) -> np.ndarray:
        """
        Compose a streamable per-residue metric from per-column statistics.

        Parameters
        ----------
        data : numpy.ndarray
            Condensed pair array with shape (n_frames, n_pairs)
        columns_by_residue : list
            Column indices per residue, from _residue_column_map
        metric : str
            One of 'mean', 'std', 'variance', 'min', 'max', 'sum', 'range'
        chunk_size : int, optional
            Number of frames per block
        use_memmap : bool
            Force block-wise processing

        Returns
        -------
        numpy.ndarray
            Metric value per residue

        Raises
        ------
        ValueError
            If metric is not supported
        """
        if metric not in CalculatorStatHelper._PER_RESIDUE_FOLDS:
            raise ValueError(
                f"Unknown per-residue metric: {metric}. Supported: "
                f"{sorted(CalculatorStatHelper._PER_RESIDUE_FOLDS) + ['median']}"
            )
        needs, fold = CalculatorStatHelper._PER_RESIDUE_FOLDS[metric]
        stats = {
            name: CalculatorStatHelper.compute_reduction_per_feature(
                data, name, chunk_size, use_memmap
            )
            for name in needs
        }
        return np.array(
            [fold(cols, stats) if cols.size else 0.0 for cols in columns_by_residue]
        )

    @staticmethod
    def _per_residue_median(data: np.ndarray, columns_by_residue: list) -> np.ndarray:
        """
        Compute the median over each residue's partner columns.

        Median cannot be composed from per-column summaries, nor streamed over
        frames, since every value is needed at once to sort. So a residue's
        columns are gathered in full, one residue at a time — this is the only
        per-residue metric not bounded by chunk_size, holding
        ``n_frames * n_partner_columns`` values per residue. The gather keeps the
        input dtype (median is order-based, so float32 stays float32).

        Parameters
        ----------
        data : numpy.ndarray
            Condensed pair array with shape (n_frames, n_pairs)
        columns_by_residue : list
            Column indices per residue, from _residue_column_map

        Returns
        -------
        numpy.ndarray
            Median per residue
        """
        result = np.zeros(len(columns_by_residue), dtype=float)
        for residue, cols in enumerate(columns_by_residue):
            if cols.size == 0:
                continue
            result[residue] = float(np.median(data[:, cols]))
        return result

    # ===== BASIC STATISTICAL METHODS =====

    @staticmethod
    def compute_differences(
        array1: np.ndarray,
        array2: np.ndarray,
        chunk_size: int = 2000,
        use_memmap: bool = False,
        preprocessing_func: Optional[callable] = None,
        **func_kwargs: Any
    ) -> np.ndarray:
        """
        Compute differences between two feature arrays with optional preprocessing.

        Parameters
        ----------
        array1 : np.ndarray
            First feature array
        array2 : np.ndarray
            Second feature array
        chunk_size : int, optional
            Chunk size for memory-mapped processing (over pairs, not frames)
        use_memmap : bool, default=False
            Whether output is for memory-mapped arrays (enables intelligent chunking)
        preprocessing_func : callable, optional
            Function to apply before computing differences (default: mean per pair)
        func_kwargs : dict
            Additional arguments for preprocessing function

        Returns
        -------
        np.ndarray
            Element-wise differences between preprocessed arrays

        Examples
        --------
        >>> array1 = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        >>> array2 = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        >>> CalculatorStatHelper.compute_differences(array1, array2)
        array([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
        """
        if preprocessing_func is None:

            def preprocessing_func(arr: np.ndarray, **kw: Any) -> np.ndarray:
                """Apply default preprocessing using mean per feature."""
                return CalculatorStatHelper.compute_func_per_feature(
                    arr, np.mean, use_memmap=use_memmap, **kw
                )

        # Apply preprocessing function to both arrays
        processed1 = preprocessing_func(array1, chunk_size=chunk_size, **func_kwargs)
        processed2 = preprocessing_func(array2, chunk_size=chunk_size, **func_kwargs)
        return processed1 - processed2

    @staticmethod
    def compute_func_per_feature(
        array: np.ndarray, 
        func: callable, 
        chunk_size: int = 2000, 
        use_memmap: bool = False, 
        **func_kwargs: Any
    ) -> np.ndarray:
        """
        Apply statistical function per feature across all frames (2D format).

        For reductions that need every frame of a column at once (median, MAD,
        percentiles) the frame axis cannot be chunked, so the feature axis is
        chunked instead. The block width is derived from ``chunk_size`` rather
        than used as-is: ``chunk_size`` counts frames, and spending it directly
        as a column count would hold ``n_frames x chunk_size`` values instead of
        the ``chunk_size x n_features`` it promises.

        Prefer ``compute_reduction_per_feature`` where the reduction can be
        streamed; it walks the frame axis and reads sequentially.

        Parameters
        ----------
        array : np.ndarray
            Feature array (NxMxM square or NxP condensed format)
        func : callable
            NumPy function to apply (np.median, np.percentile, ...)
        chunk_size : int, optional
            Number of frames per chunk; converted to a feature-block width
        use_memmap : bool, default=False
            Whether output is for memory-mapped arrays (enables intelligent chunking)
        func_kwargs : dict
            Additional arguments for the function

        Returns
        -------
        np.ndarray
            Statistical values per feature (preserves spatial dimensions)

        Examples
        --------
        >>> array = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        >>> CalculatorStatHelper.compute_func_per_feature(array, np.mean)
        array([3.0, 5.0, 7.0])
        """
        spatial_shape = array.shape[1:]
        flat_array = array.reshape(array.shape[0], -1) if array.ndim > 2 else array
        if flat_array.shape[1] == 0:
            return np.array([])
        result = CalculatorStatHelper._reduce_feature_blocks(
            flat_array, func, chunk_size, use_memmap, func_kwargs
        )
        return result.reshape(spatial_shape)

    @staticmethod
    def _resolve_feature_block_size(
        array: np.ndarray, chunk_size: Optional[int], use_memmap: bool
    ) -> int:
        """
        Decide how many features one block holds.

        Parameters
        ----------
        array : np.ndarray
            Flattened feature array with shape (n_frames, n_features)
        chunk_size : int, optional
            Requested frames per chunk
        use_memmap : bool
            Force block-wise processing

        Returns
        -------
        int
            Features per block, at least 1 and at most n_features
        """
        n_features = array.shape[1]
        should_chunk = use_memmap or FeatureShapeHelper.is_memmap(array)
        if not should_chunk:
            return n_features
        return CalculatorStatHelper.resolve_output_block_size(
            chunk_size, array.shape[0], n_features
        )

    @staticmethod
    def _reduce_feature_blocks(
        array: np.ndarray,
        func: Callable,
        chunk_size: Optional[int],
        use_memmap: bool,
        func_kwargs: dict,
    ) -> np.ndarray:
        """
        Apply func to one block of feature columns at a time.

        Every value depends only on its own column, so splitting the feature axis
        returns exactly what reducing all columns at once would.

        Parameters
        ----------
        array : np.ndarray
            Flattened feature array with shape (n_frames, n_features)
        func : callable
            Reduction applied to each block along axis 0
        chunk_size : int, optional
            Requested frames per chunk; converted to a block width
        use_memmap : bool
            Force block-wise processing
        func_kwargs : dict
            Additional arguments for the function

        Returns
        -------
        np.ndarray
            Reduction value per feature
        """
        n_features = array.shape[1]
        block = CalculatorStatHelper._resolve_feature_block_size(
            array, chunk_size, use_memmap
        )
        is_memmap_input = MemmapUtils.is_memmap_view(array)
        result_chunks = []
        for start in ProgressUtils.iterate(
            range(0, n_features, block),
            desc="Computing statistics per feature",
            unit="chunks",
        ):
            columns = array[:, start : min(start + block, n_features)]
            if is_memmap_input:
                # A column slice of a row-major memmap is strided; pull the block
                # into RAM once instead of letting func fault across the file.
                columns = np.ascontiguousarray(columns)
            result_chunks.append(func(columns, axis=0, **func_kwargs))
        return np.concatenate(result_chunks)

    @staticmethod
    def compute_func_per_frame(array: np.ndarray, chunk_size: int = 2000, use_memmap: bool = False, func: Optional[Callable] = None) -> np.ndarray:
        """
        Apply statistical function per frame across all pairs.

        Parameters
        ----------
        array : np.ndarray
            Feature array to process
        chunk_size : int, optional
            Number of frames to process per chunk
        use_memmap : bool, default=False
            Whether output is for memory-mapped arrays (enables intelligent chunking)
        func : callable, optional
            Function to apply (default: np.mean)

        Returns
        -------
        np.ndarray
            Statistical values per frame

        Examples
        --------
        >>> array = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        >>> CalculatorStatHelper.compute_func_per_frame(array, np.mean)
        """
        if func is None:
            func = np.mean

        # Intelligent chunking decision: use chunking if use_memmap=True OR input data is memmap
        should_use_chunking = use_memmap or FeatureShapeHelper.is_memmap(array)

        if not should_use_chunking:
            return CalculatorStatHelper._compute_frames_direct(array, func)
        else:
            return CalculatorStatHelper._compute_frames_chunked(array, func, chunk_size)

    @staticmethod
    def _compute_frames_direct(array: np.ndarray, func: Callable) -> np.ndarray:
        """
        Compute function per frame without chunking.

        Parameters
        ----------
        array : np.ndarray
            Feature array to process
        func : callable, optional
            Function to apply (default: np.mean)

        Returns
        -------
        np.ndarray
            Statistical values per frame
        """
        if len(array.shape) == 3:
            return func(array.reshape(array.shape[0], -1), axis=1)
        else:
            return func(array, axis=1)

    @staticmethod
    def _compute_frames_chunked(array: np.ndarray, func: Callable, chunk_size: int) -> np.ndarray:
        """
        Compute function per frame with chunking.

        Parameters
        ----------
        array : np.ndarray
            Feature array to process
        func : callable, optional
            Function to apply (default: np.mean)
        chunk_size : int, optional
            Number of frames to process per chunk

        Returns
        -------
        np.ndarray
            Statistical values per frame
        """
        result_chunks = []
        is_memmap_input = MemmapUtils.is_memmap_view(array)
        if is_memmap_input:
            ResourceUtils.tune_memmap(array, "sequential")
        for i in ProgressUtils.iterate(
            range(0, array.shape[0], chunk_size),
            desc="Computing statistics per frame",
            unit="chunks",
        ):
            end_idx = min(i + chunk_size, array.shape[0])
            chunk_result = CalculatorStatHelper._process_frame_chunk(
                array, func, i, end_idx
            )
            result_chunks.append(chunk_result)
        if is_memmap_input:
            ResourceUtils.tune_memmap(array, "random")
        return np.concatenate(result_chunks)

    @staticmethod
    def _process_frame_chunk(array: np.ndarray, func: Callable, start_idx: int, end_idx: int) -> np.ndarray:
        """
        Process a single frame chunk.

        Parameters
        ----------
        array : np.ndarray
            Feature array to process
        func : callable, optional
            Function to apply (default: np.mean)
        start_idx : int
            Start index of the chunk
        end_idx : int
            End index of the chunk

        Returns
        -------
        np.ndarray
            Statistical values per frame
        """
        if len(array.shape) == 3:
            return func(
                array[start_idx:end_idx].reshape(end_idx - start_idx, -1), axis=1
            )
        else:
            return func(array[start_idx:end_idx], axis=1)

    @staticmethod
    def compute_transitions_within_lagtime(
        array: np.ndarray, 
        threshold: float = 1.0, 
        lag_time: int = 1, 
        chunk_size: int = 2000, 
        use_memmap: bool = False
    ) -> np.ndarray:
        """
        Count transitions using lag time analysis.

        Parameters
        ----------
        array : np.ndarray
            Feature array to analyze
        threshold : float, default=1.0
            Threshold for detecting transitions
        lag_time : int, default=1
            Number of frames to look ahead
        chunk_size : int, optional
            Chunk size for processing
        use_memmap : bool, default=False
            Whether output is for memory-mapped arrays (enables intelligent chunking)

        Returns
        -------
        numpy.ndarray
            Transition counts per pair
        """
        return CalculatorStatHelper._compute_transitions_unified(
            array, threshold, lag_time, chunk_size, use_memmap, mode="lagtime"
        )

    @staticmethod
    def compute_transitions_within_window(
        array: np.ndarray, 
        threshold: float = 1.0, 
        window_size: int = 10, 
        chunk_size: int = 2000, 
        use_memmap: bool = False
    ) -> np.ndarray:
        """
        Count transitions using sliding window analysis.

        Parameters
        ----------
        array : np.ndarray
            Feature array to analyze
        threshold : float, default=1.0
            Threshold for detecting transitions
        window_size : int, default=10
            Size of sliding window
        chunk_size : int, optional
            Chunk size for processing
        use_memmap : bool, default=False
            Whether output is for memory-mapped arrays (enables intelligent chunking)

        Returns
        -------
        numpy.ndarray
            Transition counts per pair
        """
        return CalculatorStatHelper._compute_transitions_unified(
            array, threshold, window_size, chunk_size, use_memmap, mode="window"
        )

    @staticmethod
    def compute_pooled_transitions(
        segments: list,
        threshold: float,
        window_size: int,
        chunk_size: int,
        use_memmap: bool = False,
        mode: str = "lagtime",
    ) -> tuple:
        """
        Compute pooled transitions across multiple segments with boundary safety.

        Parameters
        ----------
        segments : list
            List of (n_frames, n_features) arrays
        threshold : float
            Transition threshold
        window_size : int
            Window or lag size
        chunk_size : int
            Chunk size for processing
        use_memmap : bool, default=False
            Whether to enable chunked memmap processing
        mode : str, default='lagtime'
            Computation mode ('lagtime' or 'window')

        Returns
        -------
        tuple
            (total_transitions, total_possible)
        """
        total_transitions = None
        total_possible = 0
        if not segments:
            return np.array([]), 0

        for segment in segments:
            n_frames = segment.shape[0]
            max_possible = (
                n_frames - window_size
                if mode == "lagtime"
                else n_frames - window_size + 1
            )
            if max_possible <= 0:
                continue
            transitions = CalculatorStatHelper._compute_transitions_unified(
                segment,
                threshold,
                window_size,
                chunk_size,
                use_memmap,
                mode=mode,
            ).astype(float)
            total_possible += max_possible
            if total_transitions is None:
                total_transitions = transitions
            else:
                total_transitions += transitions

        if total_transitions is None:
            total_transitions = np.zeros(segments[0].shape[1], dtype=float)

        return total_transitions, total_possible

    @staticmethod
    def compute_pooled_stability(
        segments: list,
        threshold: float,
        window_size: int,
        chunk_size: int,
        use_memmap: bool = False,
        mode: str = "lagtime",
    ) -> np.ndarray:
        """
        Compute pooled stability across segments.

        Parameters
        ----------
        segments : list
            List of (n_frames, n_features) arrays
        threshold : float
            Transition threshold
        window_size : int
            Window or lag size
        chunk_size : int
            Chunk size for processing
        use_memmap : bool, default=False
            Whether to enable chunked memmap processing
        mode : str, default='lagtime'
            Computation mode ('lagtime' or 'window')

        Returns
        -------
        numpy.ndarray
            Pooled stability values per feature
        """
        transitions, total_possible = CalculatorStatHelper.compute_pooled_transitions(
            segments,
            threshold,
            window_size,
            chunk_size,
            use_memmap,
            mode=mode,
        )
        if transitions.size == 0:
            return transitions
        if total_possible == 0:
            return np.ones_like(transitions, dtype=float)
        return 1.0 - (transitions / total_possible)

    @staticmethod
    def _compute_transitions_unified(
        array: np.ndarray, 
        threshold: float, 
        window_size: int, 
        chunk_size: int, 
        use_memmap: bool = False, 
        mode: str = "lagtime"
    ) -> np.ndarray:
        """
        Compute transitions using unified internal method.

        Parameters
        ----------
        array : np.ndarray
            Feature array
        threshold : float
            Transition threshold
        window_size : int
            Window or lag size
        chunk_size : int or None
            Chunk size for processing
        use_memmap : bool, default=False
            Whether output is for memory-mapped arrays (enables intelligent chunking)
        mode : str
            Computation mode ('lagtime' or 'window')

        Returns
        -------
        numpyp.ndarray
            Transition counts per pair
        """
        if len(array.shape) == 3:
            output_shape = (array.shape[1], array.shape[2])
            flat_array = array.reshape(array.shape[0], -1)
        else:
            output_shape = (array.shape[1],)
            flat_array = array

        result = np.zeros(output_shape, dtype=float)
        CalculatorStatHelper._compute_transitions_chunks(
            flat_array, threshold, window_size, chunk_size, use_memmap, mode, result
        )
        return result

    @staticmethod
    def _compute_transitions_chunks(
        array: np.ndarray,
        threshold: float,
        window_size: int,
        chunk_size: Optional[int],
        use_memmap: bool,
        mode: str,
        result: np.ndarray,
    ) -> None:
        """
        Compute transitions one block of feature columns at a time.

        Transitions depend on frame order, so the frame axis cannot be split and
        the feature axis is chunked instead. The block width is derived from
        ``chunk_size``, which counts frames; spending it directly as a column
        count would hold ``n_frames x chunk_size`` values per block.

        Parameters
        ----------
        array : np.ndarray
            Flattened feature array
        threshold : float
            Transition threshold
        window_size : int
            Window or lag size
        chunk_size : int, optional
            Number of frames per chunk; converted to a feature-block width
        use_memmap : bool
            Force block-wise processing
        mode : str
            Computation mode
        result : np.ndarray
            Output array to fill

        Returns
        -------
        None
            Modifies result array in-place
        """
        n_features = array.shape[1]
        block = CalculatorStatHelper._resolve_feature_block_size(
            array, chunk_size, use_memmap
        )
        flat_result = result.flatten()
        is_memmap_input = MemmapUtils.is_memmap_view(array)
        for start in ProgressUtils.iterate(
            range(0, n_features, block),
            desc="Computing transitions",
            unit="chunks",
        ):
            chunk = array[:, start : min(start + block, n_features)]
            if is_memmap_input:
                # Column slices of a row-major memmap are strided; materialise the
                # block once instead of faulting across the file per column.
                chunk = np.ascontiguousarray(chunk)
            CalculatorStatHelper._process_chunk_transitions(
                chunk, threshold, window_size, mode, flat_result, start
            )
        result[:] = flat_result.reshape(result.shape)

    @staticmethod
    def _process_chunk_transitions(
        chunk: np.ndarray, 
        threshold: float, 
        window_size: int, 
        mode: str, 
        flat_result: np.ndarray, 
        start_idx: int
    ) -> None:
        """
        Process transitions for a single chunk.

        Parameters
        ----------
        chunk : np.ndarray
            Chunk of feature array
        threshold : float
            Transition threshold
        window_size : int
            Window or lag size
        mode : str
            Computation mode
        flat_result : np.ndarray
            Flattened result array
        start_idx : int
            Start index of the chunk

        Returns
        -------
        None
            Modifies flat_result array in-place
        """
        for j in range(chunk.shape[1]):
            if mode == "lagtime":
                flat_result[start_idx + j] = (
                    CalculatorStatHelper._compute_lagtime_transitions(
                        chunk[:, j], threshold, window_size
                    )
                )
            else:
                flat_result[start_idx + j] = (
                    CalculatorStatHelper._compute_window_transitions(
                        chunk[:, j], threshold, window_size
                    )
                )

    @staticmethod
    def _compute_lagtime_transitions(data_column: np.ndarray, threshold: float, window_size: int) -> int:
        """
        Compute lagtime transitions for a single data column.

        Parameters
        ----------
        data_column : np.ndarray
            Data column to process
        threshold : float
            Transition threshold
        window_size : int
            Window or lag size

        Returns
        -------
        int
            Number of transitions
        """
        data_column = data_column.astype(float)
        diff = np.abs(data_column[:-window_size] - data_column[window_size:])
        return np.sum(diff >= threshold)

    @staticmethod
    def _compute_window_transitions(data_column: np.ndarray, threshold: float, window_size: int) -> int:
        """
        Compute window transitions for a single data column.

        Parameters
        ----------
        data_column : np.ndarray
            Data column to process
        threshold : float
            Transition threshold
        window_size : int
            Window or lag size

        Returns
        -------
        int
            Number of transitions
        """
        data_column = data_column.astype(float)
        transitions = 0
        for k in range(len(data_column) - window_size + 1):
            window_data = data_column[k : k + window_size]
            window_min = np.min(window_data)
            window_max = np.max(window_data)
            if (window_max - window_min) >= threshold:
                transitions += 1
        return transitions

    @staticmethod
    def compute_stability(
        array: np.ndarray,
        threshold: float = 2.0,
        window_size: int = 1,
        chunk_size: int = 2000,
        use_memmap: bool = False,
        mode: str = "lagtime",
    ) -> np.ndarray:
        """
        Calculate stability (inverse of transition rate) per pair.

        Parameters
        ----------
        array : np.ndarray
            Feature array to analyze
        threshold : float, default=2.0
            Threshold for stability detection
        window_size : int, default=1
            Window size for calculation
        chunk_size : int, optional
            Chunk size for processing
        use_memmap : bool, default=False
            Whether output is for memory-mapped arrays (enables intelligent chunking)
        mode : str, default='lagtime'
            Calculation mode ('lagtime' or 'window')

        Returns
        -------
        numpy.ndarray
            Stability values per pair (0=unstable, 1=stable)
        """
        transitions = CalculatorStatHelper._compute_transitions_unified(
            array, threshold, window_size, chunk_size, use_memmap, mode
        )
        max_possible_transitions = (
            array.shape[0] - window_size
            if mode == "lagtime"
            else array.shape[0] - window_size + 1
        )
        return 1.0 - (transitions / max_possible_transitions)
