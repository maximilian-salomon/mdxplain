# TODOs before Version 1.0
## Must Have

- Add_labels should work for a trajectory selection, cause the nomenclature, especially the consensus is very probable changing between different systems

- Features, Decompositions, Clustering, FeatureImportance have different Methods. They need a Type Object directly initialised. But this is complicated, because the user needs to import more than just the PipelineManager. Therefore it should exist something like a factory for these objects, which can then be used by the TypeManager

- The Data-Selector and the ComparisonData have different Modes => This should be addressable with ENUMS => Because this makes it easier. => But also for this we need a solution to handle this easily in pipeline-mode

- FeatureImportance should get 1 or 2 more Methods like RandomForest for sure and maybe SVMs

- Features should get more features: TorsionAngles (All), SASA, DSSP, Positions, CCI (a normalized combination of contacts and distances)

- Errors and prints should work with line breaks

- The K in Contacts or Distances is for maintaining neighbors => At transitions between chains or entire trajectories we have the problem that neighbors are removed that are not actually neighbors in terms of the backbone => We should fix this

- We need the visualization module and the analysis module => For this we also need requirements

- The requirements.txt should be split into dev and normal

- We need unit tests, example notebooks and documentation

- Cleanup:
Helper are helper. Sort a bit for dask-helper for example ...

## Maybes

- A selection with consensus in atom_select would be nice to have
- In the nomenclature for consensus, fragments and nomenclature could be guessed with mdciao => This would simplify the use of consensus