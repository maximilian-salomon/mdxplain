Saving and Loading
==================

The ``PipelineManager`` class provides methods to save and load the entire pipeline state,
including trajectories, features, selections, decompositions, clustering results, and analyses.

Some calculations take considerable time, so saving the pipeline allows you to avoid recomputing.

**Best Practice**: Save the pipeline after major steps in your analysis to preserve progress.

.. code:: python
    
    # Save complete pipeline state
    pipeline.save_to_single_file("my_analysis.pkl")

    # Load saved pipeline
    loaded_pipeline = PipelineManager()
    loaded_pipeline.load_from_single_file("my_analysis.pkl")

    # Print comprehensive pipeline information
    loaded_pipeline.print_info()