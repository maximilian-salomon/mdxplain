Comparative Analysis and Feature Importance
===========================================

.. todo: currently too brief; expand content.

mdxplain trains a classifier on a **comparison** -- one group of frames versus
another group (or one-vs-rest) -- to rank which molecular features best
distinguish those groups. The groups come from any data selectors, whether
cluster-based or defined by hand. Besides the single
**decision tree**, a **random forest** is available with two importance methods:
**GINI** (impurity decrease across the trees, effectively free) and **SHAP** (mean
absolute SHAP value per feature, more faithful but heavier). The forest defaults to
many shallow trees (``max_depth=6``, ``class_weight="balanced"``), which keeps SHAP
affordable and spreads the importance across correlated contacts instead of piling
it onto a single feature.

Contact features are highly redundant -- one physical coupling appears as many
near-identical neighbour pairs. ``pipeline.feature_importance.filter_importance(...)``
collapses this: it keeps only long-range pairs, then merges near neighbours
(chain-aware) into the strongest representative and records how many were merged,
shown as ``(N)`` next to the feature. The filter is non-destructive -- it writes a
filtered clone under a new name and leaves the original analysis intact. See
:doc:`basic_usage_examples/comparative_analysis_and_feature_importance` for the
full workflow.

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

    # Random forest feature importance (GINI, impurity-based)
    pipeline.feature_importance.add.random_forest(
        comparison_name="cluster_comparison",
        analysis_name="forest_gini",
        n_estimators=200,
        importance_method="gini",
        random_state=42
    )

    # Random forest feature importance (SHAP values)
    pipeline.feature_importance.add.random_forest(
        comparison_name="cluster_comparison",
        analysis_name="forest_shap",
        n_estimators=200,
        importance_method="shap",
        random_state=42
    )

    # Get top discriminative features
    top_features = pipeline.feature_importance.get_top_features(
        analysis_name="forest_gini",
        comparison_identifier="cluster_0_vs_rest",
        n=5
    )

    # Bar plot of the importance scores. Error bars show the spread of the
    # importance: across the trees for the GINI random forest, and across the
    # frames for SHAP. A single decision tree has no spread.
    pipeline.plots.feature_importance.importance_bars(
        feature_importance_name="forest_gini",
        n_top=10
    )


Reducing redundant features
---------------------------

Contact features are highly redundant: a single physical coupling shows up in
the raw features as dozens of almost-identical neighbour pairs. ``filter_importance``
collapses these into one representative per coupling. It is **non-destructive** --
the original analysis stays untouched and a filtered clone is stored under a new
name (merged features are set to zero rather than removed, so the feature index
space stays aligned).

For residue-pair features (contacts/distances) it first keeps only long-range
pairs -- residues at least ``min_sequence_separation`` positions apart *within
the same chain* -- discarding trivial within-helix contacts, for example to not
have all 4 residues of the same helix turn be the top features. It then walks the
features from strongest to weakest and merges every near-identical neighbour
(both ends within ``merge_radius`` residues, with the two ends interchangeable)
into the strongest one, counting how many were merged. For example to not have 
four separate features for the esidues of the same region in contact. 
For example ALA1-ALA2, ALA2-ALA3, ALA3-ALA4, ALA4-ALA5 are 4 features for the same contact region. 
These will be merged into one feature. All sequence logic is
chain-aware: it never merges or measures distance across a chain break.
Single-residue features (e.g. torsions) are only deduplicated.

.. code:: python

    pipeline.feature_importance.filter_importance(
        source_name="forest_shap",
        filtered_name="forest_shap_filtered",
        min_sequence_separation=20,   # ~5 helix turns; drop trivial local contacts
        merge_radius=5,               # neighbours within 5 residues are one event
    )

    # Print / plot the filtered analysis. The "(N)" after a feature shows how
    # many neighbours were merged into that representative.
    pipeline.feature_importance.print_top_n_features(
        "forest_shap_filtered", n=5
    )
    pipeline.plots.feature_importance.importance_bars("forest_shap_filtered")