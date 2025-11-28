Learn mdxplain
==============

mdxplain provides a **PipelineManager** as the central entry point for all molecular
dynamics trajectory analysis. The architecture follows a **builder pattern**, where
complex analyses are constructed step-by-step through a fluent, manager-based interface.

Key Design Principles
---------------------

- **PipelineManager**: Single entry point that coordinates all analysis operations
- **Manager-based Architecture**: Specialized managers for trajectories, features,
  clustering, decomposition, etc.
- **Pipeline Data**: Central data structure (`pipeline.data`) that accumulates all
  analysis results
- **Fluent API**: Intuitive, chainable methods like `pipeline.feature.add.contacts()`

Basic Usage Examples
--------------------

.. todo: streamline existing base examples for clarity and efficiency

.. toctree::
   :maxdepth: 1

   basic_usage_examples/quick_start_example
   basic_usage_examples/memory-efficient_processing
   basic_usage_examples/trajectory_management
   basic_usage_examples/feature_computation
   basic_usage_examples/feature_selection
   basic_usage_examples/feature_reduction
   basic_usage_examples/dimensionality_reduction
   basic_usage_examples/clustering
   basic_usage_examples/structural_analysis
   basic_usage_examples/feature_statistics
   basic_usage_examples/data_selection
   basic_usage_examples/comparative_analysis_and_feature_importance
   basic_usage_examples/plotting
   basic_usage_examples/saving_and_loading

Tutorials
---------

Here's a complete conformational analysis workflow:

.. placeholder for notebooks

.. toctree::
   :maxdepth: 1

   placeholder

.. toctree notebooks end