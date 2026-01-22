Performance Settings (Quick Guide)
==================================

mdxplain exposes a small set of stability-focused resource controls through
`pipeline.config.performance`. These settings are applied immediately when you
change them.

Key settings
------------

- `use_stability_config` (PipelineManager init):
  - `None` (default): apply stability settings only when `use_memmap=True`
  - `True`: always apply stability settings
  - `False`: keep OS defaults unless you change the config manually
- `pipeline.config.performance.resource_nice`:
  lower CPU priority to keep the system responsive
- `pipeline.config.performance.resource_io_priority`:
  reduce I/O priority to avoid starving the OS
- `pipeline.config.performance.resource_cpu_affinity`:
  keep a few cores free for the system
- `pipeline.config.performance.auto_blas_thread_limit`:
  prevent BLAS/OpenMP oversubscription

Examples
--------

Enable stability settings explicitly:

.. code-block:: python

   pipeline = PipelineManager(use_stability_config=True)

Change resource limits at runtime:

.. code-block:: python

   pipeline.config.performance.update(
       resource_nice=10,
       resource_io_priority="low",
       reserve_cores=2,
   )

You can also update a single field directly:

.. code-block:: python

   pipeline.config.performance.reserve_cores = 1

For the full rationale and OS-specific behavior, see
:doc:`../performance_and_system_stability`.
