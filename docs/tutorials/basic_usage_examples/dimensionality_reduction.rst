Dimensionality Reduction
========================

.. todo: requires more explanation:
    Include prerequisites (e.g., output from feature selector for selection_name).

Four decomposition methods are available:

- PCA
- Kernel PCA
- Contact Kernel PCA
- Diffusion Maps

.. code:: python

    # Standard PCA
    pipeline.decomposition.add.pca(
        n_components=10, selection_name="my_selection"
    )

    # Kernel PCA with RBF kernel
    pipeline.decomposition.add.kernel_pca(
        n_components=10, kernel='rbf', gamma=0.01,
        selection_name="my_selection"
    )

    # Contact Kernel PCA (optimized for binary contact data)
    pipeline.decomposition.add.contact_kernel_pca(
        n_components=10, gamma=0.001, selection_name="contacts_only"
    )

    # Diffusion Maps
    pipeline.decomposition.add.diffusion_maps(
        n_components=10, selection_name="my_selection"
    )


Reducing components after the fact
----------------------------------

PCA and Kernel PCA order their components by descending variance, so the
leading ``n`` components of a decomposition are the same ones a fresh run with
``n_components=n`` would produce. ``reduce_components`` uses this to keep only
the first ``n`` components of an existing decomposition by slicing the stored
transformed data -- it does not recompute the eigendecomposition, which is the
expensive step. This is useful when a decomposition was auto-selected to more
components than a downstream step needs.

The operation is non-destructive: the original decomposition is left untouched
and the truncated result is stored under a new name.

.. code:: python

    # A decomposition auto-selected 30 components; keep the leading 5
    pipeline.decomposition.reduce_components(
        source_name="ContactKernelPCA",
        new_name="ContactKernelPCA_5",
        n_components=5,
    )

    # Use the reduced decomposition downstream like any other
    pipeline.clustering.add.dpa("ContactKernelPCA_5", Z=2.5)