Dimensionality Reduction
========================

Four decomposition methods are available:

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