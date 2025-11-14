Quick Start Example
===================

See :doc:`Complete Workflow Example <../00_introduction>` for a detailed
walkthrough of this workflow.

Here's a complete conformational analysis workflow:

.. code:: python

    from mdxplain import PipelineManager
    
    # Initialize pipeline
    pipeline = PipelineManager(use_memmap=True, chunk_size=1000)

    # Load trajectory and add residue labels
    pipeline.trajectory.load_trajectories("data/2RJY/")
    pipeline.trajectory.add_labels(traj_selection="all")

    # Compute features
    pipeline.feature.add.distances()
    pipeline.feature.add.contacts(cutoff=4.5)

    # Create feature selection
    pipeline.feature_selector.create("contacts_only")
    pipeline.feature_selector.add.contacts("contacts_only", "all")
    pipeline.feature_selector.select("contacts_only")

    # Dimensionality reduction
    pipeline.decomposition.add.contact_kernel_pca(
        n_components=10, gamma=0.001, selection_name="contacts_only"
    )

    # Clustering
    pipeline.clustering.add.dpa(selection_name="ContactKernelPCA", Z=2.0)

    # Create data selectors for each cluster
    n_clusters = pipeline.data.cluster_data["DPA"].get_n_clusters()
    for i in range(n_clusters):
        pipeline.data_selector.create(f"cluster_{i}")
        pipeline.data_selector.select_by_cluster(f"cluster_{i}", "DPA", [i])

    # Feature importance analysis
    cluster_names = [f"cluster_{i}" for i in range(n_clusters)]
    pipeline.comparison.create_comparison(
        name="cluster_comparison", mode="one_vs_rest",
        feature_selector="contacts_only",
        data_selectors=cluster_names
    )
    pipeline.feature_importance.add.decision_tree(
        comparison_name="cluster_comparison", max_depth=3
    )

    # Get top discriminative features
    top_features = pipeline.feature_importance.get_top_features(
        analysis_name="feature_importance",
        comparison_identifier="cluster_0_vs_rest", n=5
    )

    # Save complete analysis
    pipeline.save_to_single_file("my_analysis.pkl")