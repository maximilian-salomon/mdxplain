# mdxplain Orchestration

This table provides an overview of the methods, algorithms, and backends used in mdxplain, organized by category.

<style>
table.docutils {
    width: 100%;
    max-width: 100%;
    font-size: 0.9em;
}
table.docutils th,
table.docutils td {
    word-wrap: break-word;
    overflow-wrap: break-word;
    white-space: normal;
    vertical-align: top;
    padding: 8px 6px;
}
table.docutils th:nth-child(1),
table.docutils td:nth-child(1) {
    width: 8%;
    max-width: 130px;
}
table.docutils th:nth-child(2),
table.docutils td:nth-child(2) {
    width: 18%;
}
table.docutils th:nth-child(3),
table.docutils td:nth-child(3) {
    width: 16%;
}
table.docutils th:nth-child(4),
table.docutils td:nth-child(4) {
    width: 58%;
}
table.docutils td {
    word-wrap: break-word;
    overflow-wrap: break-word;
    hyphens: auto;
}
table.docutils td[rowspan] {
    border-right: 1px solid #e1e4e5 !important;
}
table.docutils tbody tr td.category-bg-even {
    background-color: #f3f6f6 !important;
}
table.docutils tbody tr td.category-bg-odd {
    background-color: transparent !important;
}
table.docutils tbody tr.row-category-even td:not([rowspan]) {
    background-color: #f3f6f6 !important;
}
table.docutils tbody tr.row-category-odd td:not([rowspan]) {
    background-color: transparent !important;
}
table.docutils tbody tr.row-alt-even td:not([rowspan]) {
    background-color: #f3f6f6 !important;
}
table.docutils tbody tr.row-alt-odd td:not([rowspan]) {
    background-color: transparent !important;
}
</style>

<table class="docutils align-default">
<thead>
<tr class="row-odd">
<th class="head"><p>Category</p></th>
<th class="head"><p>Method/Algorithm</p></th>
<th class="head"><p>Backend</p></th>
<th class="head"><p>Explanation</p></th>
</tr>
</thead>
<tbody>
<tr class="row-category-even">
<td class="category-bg-even" rowspan="1"><p>Trajectory</p></td>
<td><p>Trajectory handling</p></td>
<td><p>mdtraj + mdxplain</p></td>
<td><p>mdxplain: In memmap-mode uses DaskTrajectory - Dask/Zarr manages data streaming and storage.<br>
 <br>
 mdtraj: Reads topology and frames, performs geometric operations (superpose, smooth, align, distances) on each loaded chunk.<br>
 <br>
 In normal mode uses mdtraj trajectories.</p></td>
</tr>
<tr class="row-category-odd">
<td class="category-bg-odd" rowspan="2"><p>Nomenclature/Labeling</p></td>
<td><p>Consensus labeling (GPCR/CGN/KLIFS)</p></td>
<td><p>mdciao</p></td>
<td><p>Consensus labels come from mdciao; mdxplain maps them to features/selectors and integrates them into its internal metadata system.</p></td>
</tr>
<tr class="row-alt-even">
<td><p>Standard labeling and metadata generation</p></td>
<td><p>mdxplain</p></td>
<td><p>mdxplain implements own metadata / AA labeling using topology from mdtraj.</p></td>
</tr>
<tr class="row-category-even">
<td class="category-bg-even" rowspan="1"><p>Feat. Sel. DSL</p></td>
<td><p>Feature Selector/Internal DSL</p></td>
<td><p>mdxplain + mdtraj</p></td>
<td><p>Whole Selection DSL by mdxplain. For atom-selection from topology mdtraj DSL is used.</p></td>
</tr>
<tr class="row-category-odd">
<td class="category-bg-odd" rowspan="2"><p>Data Handling</p></td>
<td><p>Archiving and large data handling</p></td>
<td><p>Pickle</p></td>
<td><p>Archiving and saving implemented by mdxplain using Pickle.</p></td>
</tr>
<tr class="row-alt-even">
<td><p>Memmap arrays and low-level arithmetic</p></td>
<td><p>NumPy</p></td>
<td><p>Intermediate data is saved as NumPy memmap arrays.<br>
 <br>
 If not stated differently, NumPy also used for basic arithmetics like mean, median, var, etc.</p></td>
</tr>
<tr class="row-category-even">
<td class="category-bg-even" rowspan="10"><p>Geometric Features</p></td>
<td><p>Coordinates</p></td>
<td><p>mdxplain</p></td>
<td><p>Whole feature type by mdxplain, trajectory object provides coordinates & topology.</p></td>
</tr>
<tr class="row-alt-odd">
<td><p>Distances</p></td>
<td><p>mdtraj</p></td>
<td><p>mdxplain orchestrates uses mdtraj.compute_contacts as computation backend.</p></td>
</tr>
<tr class="row-alt-even">
<td><p>Contacts</p></td>
<td><p>mdxplain</p></td>
<td><p>Whole feature type by mdxplain, uses distance outputs (mdtraj kernel).</p></td>
</tr>
<tr class="row-alt-odd">
<td><p>Torsions</p></td>
<td><p>mdtraj</p></td>
<td><p>mdxplain orchestrates and uses mdtraj.compute_phi/psi/omega/chi as backend.</p></td>
</tr>
<tr class="row-alt-even">
<td><p>DSSP</p></td>
<td><p>mdtraj</p></td>
<td><p>mdxplain orchestrates and uses mdtraj.dssp as backend.</p></td>
</tr>
<tr class="row-alt-odd">
<td><p>SASA</p></td>
<td><p>mdtraj</p></td>
<td><p>mdxplain orchestrates and uses mdtraj.shrake_rupley as backend.</p></td>
</tr>
<tr class="row-alt-even">
<td><p>Reduction Logic</p></td>
<td><p>mdxplain + SciPy</p></td>
<td><p>mdxplain implements feature-reduction and uses NumPy/SciPy for metrics where possible, extending them when needed.</p></td>
</tr>
<tr class="row-alt-odd">
<td><p>RMSD</p></td>
<td><p>mdxplain</p></td>
<td><p>The trajectory object provides topology/coordinates; mdxplain handles computation plus data/metadata, including window and reference metrics (frame-to-frame or to-reference) and variants like RMSD with mean or median for flexible systems.</p></td>
</tr>
<tr class="row-alt-even">
<td><p>RMSF</p></td>
<td><p>mdxplain + Numba-JIT</p></td>
<td><p>Trajectory supplies topology/coords; mdxplain handles computation and metadata, including window/reference metrics and RMSF variants (mean/median).<br>
 <br>
 It also performs residue-level aggregation (mean, median, RMS, RMdS) with Numba-JIT acceleration.</p></td>
</tr>
<tr class="row-alt-odd">
<td><p>MAD</p></td>
<td><p>mdxplain</p></td>
<td><p>mdxplain offers MAD values in frame-based and atom- / residue-based manner.<br>
 <br>
 This is implemented by mdxplain directly inside the RMSD and RMSF calculators.</p></td>
</tr>
<tr class="row-category-odd">
<td class="category-bg-odd" rowspan="3"><p>Dim Reduction</p></td>
<td><p>PCA/IncrementalPCA</p></td>
<td><p>scikit-learn</p></td>
<td><p>PCA and incremental PCA wrap scikit-learn; mdxplain orchestrates.</p></td>
</tr>
<tr class="row-alt-even">
<td><p>Kernel PCA</p></td>
<td><p>scikit-learn + scipy + mdxplain</p></td>
<td><p>Standard KPCA uses scikit-learn; iterative KPCA and metadata/data handling are implemented by mdxplain.<br>
 <br>
 Nystroem KPCA combines scikit-learn's RBF Nystroem with IncrementalPCA, with mdxplain adding component/epsilon approximations.</p></td>
</tr>
<tr class="row-alt-odd">
<td><p>Diffusion Maps</p></td>
<td><p>mdxplain</p></td>
<td><p>Full algorithm incl. iterative and nystroem mode implemented by mdxplain.</p></td>
</tr>
<tr class="row-category-even">
<td class="category-bg-even" rowspan="3"><p>Clustering</p></td>
<td><p>DBSCAN</p></td>
<td><p>scikit-learn</p></td>
<td><p>mdxplain orchestrates DBSCAN: standard mode uses scikit-learn's DBSCAN, precomputed mode uses scikit-learn's NearestNeighbors, and kNN-mode combines scikit-learn's DBSCAN on a subsample with scikit-learn's kNN for assignment.</p></td>
</tr>
<tr class="row-alt-odd">
<td><p>HDBSCAN</p></td>
<td><p>hdbscan + scikit-learn</p></td>
<td><p>mdxplain orchestrates HDBSCAN workflows: standard mode uses the hdbscan implementation, approximate-prediction mode uses random sampling plus hdbscan's own prediction, and kNN-mode combines hdbscan on a subsample with scikit-learn's kNN for assignment.</p></td>
</tr>
<tr class="row-alt-even">
<td><p>DPA</p></td>
<td><p>dpa + scikit-learn</p></td>
<td><p>mdxplain orchestrates DPA workflows: standard mode uses the dpa library's DPA implementation, and kNN-mode runs DPA on a subsample and assigns remaining points via scikit-learn's kNN.</p></td>
</tr>
<tr class="row-category-odd">
<td class="category-bg-odd" rowspan="1"><p>Feature Importance</p></td>
<td><p>Decision Tree</p></td>
<td><p>scikit-learn</p></td>
<td><p>mdxplain orchestrates decision-tree workflows: it uses scikit-learn's DecisionTreeClassifier, adds stratified sampling for large datasets, and provides comparison modes (one-vs-rest, multiclass, binary, pairwise).</p></td>
</tr>
<tr class="row-category-even">
<td class="category-bg-even" rowspan="1"><p>Feature Statistics</p></td>
<td><p>Feature-type analysis</p></td>
<td><p>mdxplain</p></td>
<td><p>mdxplain implements feature-based analysis and uses NumPy/SciPy for metrics where possible, extending them when needed.</p></td>
</tr>
<tr class="row-category-odd">
<td class="category-bg-odd" rowspan="6"><p>Plots</p></td>
<td><p>DecisionTree-Plotter</p></td>
<td><p>mdxplain + matplotlib</p></td>
<td><p>Layout, styling, export by mdxplain. Uses tree-structure of scikit-learn model.<br>
 <br>
 Plot-library is matplotlib.</p></td>
</tr>
<tr class="row-alt-even">
<td><p>DensityPlotter</p></td>
<td><p>mdxplain + matplotlib + scipy</p></td>
<td><p>mdxplain implementation using matplotlib.<br>
 <br>
 DensityPlotter uses scipy KDEs to display smoothed histrograms.</p></td>
</tr>
<tr class="row-alt-odd">
<td><p>ViolinPlotter</p></td>
<td><p>mdxplain + matplotlib</p></td>
<td><p>mdxplain implementation using matplotlib.</p></td>
</tr>
<tr class="row-alt-even">
<td><p>TimeSeries-Plotter</p></td>
<td><p>mdxplain + matplotlib</p></td>
<td><p>mdxplain implementation using matplotlib.<br>
 <br>
 Time-series can be smoothed using a savgol filter from scipy.</p></td>
</tr>
<tr class="row-alt-odd">
<td><p>Landscape-Plotter</p></td>
<td><p>mdxplain + matplotlib + seaborn + scipy</p></td>
<td><p>mdxplain implementation using matplotlib and seaborn.<br>
 <br>
 Uses NumPy for histograms and SciPy for optional KDE smoothing.</p></td>
</tr>
<tr class="row-alt-even">
<td><p>Cluster-Membership-Plotter</p></td>
<td><p>mdxplain + matplotlib</p></td>
<td><p>mdxplain implementation using matplotlib.</p></td>
</tr>
<tr class="row-category-even">
<td class="category-bg-even" rowspan="2"><p>3D Viz</p></td>
<td><p>StructureViz Feature Service</p></td>
<td><p>mdxplain + mdtraj + PyMol</p></td>
<td><p>mdxplain implements representative-finding (centroid or decision-tree–based), uses mdtraj to generate PDBs.<br>
<br>
 mdxplain embed feature-importance values as B-factors, and provides PyMOL script generation for optional visualization.<br>
 <br>
 PyMOL can be used optional to visualize this script.</p></td>
</tr>
<tr class="row-alt-odd">
<td><p>NGLView</p></td>
<td><p>mdxplain + mdtraj + nglview</p></td>
<td><p>mdxplain implements representative-finding (centroid or decision-tree–based), uses mdtraj to generate PDBs.<br>
 <br>
 mdxplain embed feature-importance values as B-factors, and provides an extended nglview widget with checkboxes, selections, and legends for PyMOL-like 3D comparison in Jupyter.</p></td>
</tr>
</tbody>
</table>
