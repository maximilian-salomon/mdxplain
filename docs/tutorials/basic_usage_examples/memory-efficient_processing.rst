Memory-Efficient Processing
===========================

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

    - Example: six 3500-frame trajectories with 16 GB RAM → ``chunk_size=300``