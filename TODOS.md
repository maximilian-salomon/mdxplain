# TODOs before Version 1.0
## Must Have
- Save of PipelineManager manual

- print_trajectory_info() add to trajectoryManager

- Add print Info to all manager

- Add option to select Frames as trajectory with DataSelector

- Features, Decompositions, Clustering, FeatureImportance have different Methods. They need a Type Object directly initialised. But this is complicated, because the user needs to import more than just the PipelineManager. Therefore it should exist something like a factory for these objects, which can then be used by the TypeManager

- The Data-Selector and the ComparisonData have different Modes => This should be addressable with ENUMS => Because this makes it easier. => But also for this we need a solution to handle this easily in pipeline-mode

- Features should get more features: TorsionAngles (All), SASA, DSSP, Positions, CCI (a normalized combination of contacts and distances)

- We need the visualization module and the analysis module => For this we also need requirements

- We need unit tests, example notebooks and documentation

- Cleanup

## Maybes
- In the nomenclature for consensus, fragments and nomenclature could be guessed with mdciao => This would simplify the use of consensus

- Save of PipelineManager auto

- config

- Errors and prints should work with line breaks

- Logging!

- FeatureImportance should get 1 or 2 more Methods like RandomForest for sure and maybe SVMs
