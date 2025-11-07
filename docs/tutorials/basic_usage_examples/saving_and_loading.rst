Saving and Loading
==================

.. code:: python
    
    # Save complete pipeline state
    pipeline.save("my_analysis.pkl")

    # Load saved pipeline
    loaded_pipeline = PipelineManager()
    loaded_pipeline.load("my_analysis.pkl")

    # Print comprehensive pipeline information
    loaded_pipeline.print_info()