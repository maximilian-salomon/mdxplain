Clustering
==========

Three density-based clustering algorithms:

- DPA (Density Peak Advanced)
- DBSCAN (Density-Based Spatial Clustering of Applications with Noise)
- HDBSCAN (Hierarchical DBSCAN)

.. code:: python

    # Density Peak Advanced (DPA) - automatic cluster number detection
    # In our personal experience often good results and easy to use.
    pipeline.clustering.add.dpa(
        selection_name="ContactKernelPCA", Z=3.0
    )

    # DBSCAN
    pipeline.clustering.add.dbscan(
        selection_name="my_decomposition", eps=0.5, min_samples=5
    )

    # HDBSCAN
    pipeline.clustering.add.hdbscan(
        selection_name="my_decomposition", min_cluster_size=10
    )