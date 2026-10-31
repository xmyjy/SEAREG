Tutorial data
=============

Each single-section example includes a spot-by-peak ``counts.h5ad`` with spatial coordinates
and a peak-by-motif ``motifs.h5`` annotation matrix. Files are prepared at
startup; motif import or scanning follows peak filtering.
The repository includes these five input pairs and the four-section embryo
count matrix as lossless gzip files, plus the specified named output embeddings.
The complete GitHub clone and Download ZIP contain the actual files.
Wheel and source distributions omit tutorials, data and results. Complete model
checkpoints are not distributed. All six tutorials default to training a model
and analyzing its final exports. T6 trains the joint-section model with adaptive depth gating and
records its actual embedding source and analysis in ``tutorial_execution.json``.
Its saved example contains the final node exports, native analysis and figures
under ``result/Embryo_multislice/adaptive_depth_gate/``.
Optional provided mode reads the selected ``EMBEDDING_DIR`` without training.
The same first-cell interface accepts
your own counts, motif annotations, output directory and analysis parameters.
See :doc:`workflow`.

The developing mouse brain example (T6) includes
``data/Embryo_multislice/annotations.tsv`` and anatomy palettes.
Its 7,118-spot cohort has 7,037 known annotations and 81 Unknowns. Reference maps
appear after reading counts and before training; labels are joined by spot ID
for display. The model uses counts, shared peak identities, section IDs and
native coordinates. Palette source tables and
``annotation_palette_manuscript_source.json`` record the display provenance.
See :doc:`multislice` for the joint model, exact-K analysis and saved-result reopening.

.. list-table:: Included datasets
   :header-rows: 1
   :widths: 30 35 35

   * - Dataset
     - Sequencing platform
     - Local directory
   * - Mouse embryo (E13)
     - spatial ATAC–RNA-seq (ATAC modality)
     - ``data/ME13_joint_2187/``
   * - Mouse embryo (E13)
     - spatial-ATAC-seq
     - ``data/ME13_ATAC_2275/``
   * - Mouse brain (P22)
     - spatial ATAC–RNA-seq (ATAC modality)
     - ``data/P22_ATAC/``
   * - Mouse brain (P22)
     - spatial CUT&Tag–RNA-seq (H3K27ac modality)
     - ``data/P22_H3K27ac/``
   * - Human tonsil
     - spatial-ATAC-seq
     - ``data/Tonsil/``
   * - Developing mouse brain (E11.0–E18.5)
     - MISAR-seq (ATAC modality; four sections)
     - ``data/Embryo_multislice/``

Original sequencing studies
---------------------------

Tutorial names follow the technologies and samples described in the original
sequencing papers. The directory names above identify the included defaults;
user data does not need to be registered under one of these names.

.. list-table:: Study sources and model inputs
   :header-rows: 1
   :widths: 14 30 28 28

   * - Tutorial
     - Original study
     - Study accession
     - Modality used for SEAREG training
   * - 1
     - Zhang et al., Nature (2023)
     - GSE205055
     - ATAC, with peak–motif associations and spatial coordinates
   * - 2
     - Deng et al., Nature (2022)
     - GSE171943
     - ATAC, with peak–motif associations and spatial coordinates
   * - 3
     - Zhang et al., Nature (2023)
     - GSE205055
     - ATAC, with peak–motif associations and spatial coordinates
   * - 4
     - Zhang et al., Nature (2023)
     - GSE205055
     - CUT&Tag H3K27ac, with peak–motif associations and spatial coordinates
   * - 5
     - Deng et al., Nature (2022)
     - GSE171943
     - ATAC, with peak–motif associations and spatial coordinates
   * - 6
     - Jiang et al., Nature Methods (2023)
     - NODE OEP003285
     - ATAC, with section identities and spatial coordinates

* `Zhang et al., Spatial epigenome–transcriptome co-profiling of mammalian tissues (2023) <https://doi.org/10.1038/s41586-023-05795-1>`_;
  `GEO GSE205055 <https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE205055>`_.
* `Deng et al., Spatial profiling of chromatin accessibility in mouse and human tissues (2022) <https://doi.org/10.1038/s41586-022-05094-1>`_;
  `GEO GSE171943 <https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE171943>`_.
* `Jiang et al., Simultaneous profiling of spatial gene expression and chromatin accessibility during mouse brain development (2023) <https://doi.org/10.1038/s41592-023-01884-1>`_;
  `NODE OEP003285 <https://www.biosino.org/node/project/detail/OEP003285>`_.

The co-profiling studies measured RNA alongside chromatin. T1–T5 use the
indicated chromatin modality, motif associations and spatial coordinates;
their joint UMAPs display spot, peak and motif nodes. T6 uses ATAC counts,
shared peak identities, section IDs and spatial coordinates without motifs;
its joint UMAP contains spot and shared-peak nodes. Tutorials begin
with processed matrices; FASTQ processing and peak calling precede these inputs.

Local input and bundled output layout
-------------------------------------

Each single-section ``data/<dataset>/`` contains compressed counts and motif
matrices; ``Embryo_multislice`` contains counts without motifs.
P22_ATAC counts use numbered gzip parts. Notebooks prepare the selected counts
and optional motif files before constructing the workflow. Actual motif
processing occurs after filtering. Extraction checks the archive and
original-file identities recorded in ``data/manifest.json``.

Existing ``ATAC_PATH`` and ``MOTIF_PATH`` files are used directly and validated
by the workflow's data checks. They require no example hash or manifest entry.
Missing counts raise an error; missing motifs follow the configured read/scan
policy. Only an exact bundled path can trigger automatic extraction. To
explicitly prepare and verify all bundled inputs:

.. code-block:: bash

   python tools/prepare_data.py --dataset all

The provided spot/peak/motif exports are under
``result/<dataset>/training/embeddings/300_epoch/`` (T1–T2) or ``500_epoch/``
(T3–T5). T6's final adaptive_depth_gate spot/peak exports are at
``result/Embryo_multislice/adaptive_depth_gate/500_epoch/``. The same result directory
contains the saved ``analysis/``, ``figures/`` and ``tutorial_execution.json``.
The receipt identifies the selected counts, trained embeddings and fitted
analysis. Reopening ``analysis/`` reads its labels, graph and UMAP coordinates
without fitting.

The tutorial loader uses the selected directory's own ``export.json`` and
matching counts for validation. Both modes write new analysis under
``OUTPUT_DIR/runs/``; new training keeps its model and exports with the run.
See :doc:`workflow` for choosing a source and recovering native training.

Checkpoints, unpacked bundled inputs and generated default runs are ignored
by Git.
