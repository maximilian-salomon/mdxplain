Comparative Analysis and Feature Importance
===========================================

.. code:: python
    
    # Create data selectors for clusters
    pipeline.data_selector.create("cluster_0")
    pipeline.data_selector.select_by_cluster("cluster_0", "DPA", [0])

    # One-vs-rest comparison
    pipeline.comparison.create_comparison(
        name="cluster_comparison",
        mode="one_vs_rest",
        feature_selector="contacts_only",
        data_selectors=["cluster_0", "cluster_1", "cluster_2"]
    )

    # Decision tree feature importance
    pipeline.feature_importance.add.decision_tree(
        comparison_name="cluster_comparison",
        analysis_name="importance",
        max_depth=3
    )

    # Get top discriminative features
    top_features = pipeline.feature_importance.get_top_features(
        analysis_name="importance",
        comparison_identifier="cluster_0_vs_rest",
        n=5
    )