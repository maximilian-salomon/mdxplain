Memory-Efficient Processing
===========================

.. todo: Is it possible to adjust chunksize after pipeline creation?

For large trajectories, mdxplain supports memory-mapped processing:

.. code:: python

    # Enable memory mapping for datasets larger than RAM
    pipeline = PipelineManager(use_memmap=True, chunk_size=1000)

Memory Mapping Guidelines
-------------------------

- **Enable** for trajectories approaching/exceeding available RAM
- **Enable** when analyzing multiple large trajectories simultaneously
- **Disable** for small/medium datasets that fit in RAM (faster processing)
- **Chunk size**: Start with 2000 frames; increase if RAM allows, decrease if memory pressure occurs

    - Example: six 3500-frame trajectories with 16 GB RAM → ``chunk_size=500``

Reusing Cached Results
----------------------

With memory mapping enabled, every persistent result is written to a ``.dat``
file under the cache directory: computed features, PCA/Kernel PCA transformed
data, and cluster labels. ``reuse_memmap_cache=True`` lets a run reopen a
matching cached ``.dat`` instead of recomputing it -- useful when re-running an
analysis with the same parameters after a restart and one forgot do save the pipeline 
but the cache is persistent.

.. code:: python

    # First run: computes and caches everything (the cache is always written
    # when use_memmap=True, regardless of reuse_memmap_cache).
    pipeline = PipelineManager(use_memmap=True)
    # ... compute features / decomposition / clustering ...

    # Later run: reuse the cached results instead of recomputing them.
    pipeline = PipelineManager(use_memmap=True, reuse_memmap_cache=True)
    # ... the same steps reopen the cached .dat files ...

How reuse stays correct
~~~~~~~~~~~~~~~~~~~~~~~~~

Next to each result ``.dat``, mdxplain writes a small sidecar recording the
shape, dtype, and the parameters that define the result (for example a
contact ``cutoff``, a PCA ``n_components``, or a DBSCAN ``eps``). A cached
result is reused **only** when the sidecar exists and its parameters, dtype,
and on-disk size all match the request. This rules out two failure modes:

- a result computed with **different parameters** is never silently reused
  (change a parameter and it recomputes);
- a run aborted **mid-write** leaves no sidecar, so a partial file is never
  reused.

Notes
~~~~~

- ``reuse_memmap_cache`` defaults to ``False`` and only has an effect when
  ``use_memmap=True``.
- To force a fresh computation, pass ``force=True`` to the relevant ``add``
  call; this removes the cached files (and their sidecars) first.
- Temporary intermediates (for example Diffusion Maps kernel matrices) are
  never reused -- they are deleted after use.