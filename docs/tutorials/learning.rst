Learn mdxplain
==============

mdxplain provides a **PipelineManager** as the central entry point for all molecular dynamics trajectory analysis.
The architecture follows a **builder pattern**, where complex analyses are constructed step-by-step through a
fluent, manager-based interface.

Key Design Principles
---------------------

- **PipelineManager**: Single entry point that coordinates all analysis operations
- **Manager-based Architecture**: Specialized managers for trajectories, features, clustering, decomposition, etc.
- **Pipeline Data**: Central data structure (`pipeline.data`) that accumulates all analysis results
- **Fluent API**: Intuitive, chainable methods like `pipeline.feature.add.contacts()`

Basic Usage Examples
--------------------

Examples will be added.

Complete Workflow Example
-------------------------

Here's a complete conformational analysis workflow:

.. toctree::
   :maxdepth: 1

   00_introduction