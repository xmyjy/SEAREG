Tutorials
=========

Choose a tutorial by assay and number of sections. Each page displays the
original notebook's saved outputs, and provides that notebook for download.
Reading the website requires no scientific environment. Start with
:doc:`../quickstart` to choose between training, analyzing existing exports
and reopening saved analysis.

Choose an example
-----------------

The input directories below are relative to the complete repository checkout.
Each contains compressed ``counts.h5ad``; T1–T5 also include ``motifs.h5``.
Counts include native spatial coordinates. The co-profiled RNA modality is
not supplied to these models. See :doc:`../data_access` for studies, accessions
and extraction, and :doc:`../input_data` for the input schema.

.. list-table:: Tutorial selection
   :header-rows: 1
   :widths: 25 20 25 30

   * - Tutorial and sections
     - Model inputs
     - Bundled directory
     - Saved figures
   * - :doc:`T1: E13 mouse embryo <T1_Mouse_Embryo_E13_Spatial_ATAC_RNA>`; one section
     - ATAC + peak–motif associations + XY
     - ``data/ME13_joint_2187/``
     - Four: spatial domains, spot UMAP, joint UMAP, feature spatial scores/accessibility
   * - :doc:`T2: E13 mouse embryo <T2_Mouse_Embryo_E13_Spatial_ATAC>`; one section
     - ATAC + peak–motif associations + XY
     - ``data/ME13_ATAC_2275/``
     - Three: spatial domains, spot UMAP, joint UMAP
   * - :doc:`T3: P22 mouse brain <T3_Mouse_Brain_P22_Spatial_ATAC_RNA>`; one section
     - ATAC + peak–motif associations + XY
     - ``data/P22_ATAC/``
     - Three: spatial domains, spot UMAP, joint UMAP
   * - :doc:`T4: P22 mouse brain <T4_Mouse_Brain_P22_Spatial_CUTTag_RNA_H3K27ac>`; one section
     - CUT&Tag H3K27ac + peak–motif associations + XY
     - ``data/P22_H3K27ac/``
     - Three: spatial domains, spot UMAP, joint UMAP
   * - :doc:`T5: human tonsil <T5_Human_Tonsil_Spatial_ATAC>`; one section
     - ATAC + peak–motif associations + XY
     - ``data/Tonsil/``
     - Three: spatial domains, spot UMAP, joint UMAP
   * - :doc:`T6: developing mouse brain <T6_Mouse_Brain_Development_MISAR_seq>`; four sections
     - MISAR-seq ATAC + shared peaks + section IDs + XY; no motifs
     - ``data/Embryo_multislice/``; reference annotations and palettes for display
     - Four: reference maps, spatial domains, three-view spot UMAP, three-view joint UMAP

Modes and prerequisites
-----------------------

All six downloaded notebooks default to ``RUN_MODE="train"``. Set paths,
output base and options in the first code cell, then run cells in order.
Each mode writes a unique run directory; neither mode replaces the provided
exports. Use the :doc:`installation guide <../installation>` for the pinned
Linux/Python environment and device selection, and :doc:`../workflow` for
durable jobs and recovery.

.. list-table:: Running a downloaded notebook
   :header-rows: 1
   :widths: 20 35 45

   * - Mode
     - What it does
     - Prerequisites
   * - ``train`` (T1–T6 default)
     - Prepares a network, trains a model, then analyzes its final exports
     - Installed SEAREG tutorial environment, tmux, matching counts and configured CPU or CUDA device; T1–T5 use the supplied motif matrix unless motifs are disabled
   * - ``provided`` (T1–T6 optional)
     - Reads ``EMBEDDING_DIR`` and computes domains/UMAPs from those embeddings
     - Installed SEAREG analysis/notebook environment, matching counts and valid propagated exports with ``export.json``; no motif matrix, R, GPU, tmux or complete checkpoint required

Generating new motif annotations additionally requires the
:doc:`R scanner and reference resources <../motif_annotation>`.
Existing bundled matrices need no R scanning. Profiles select CUDA by
default; CPU configuration is described in :doc:`../workflow`.

For viewing the existing figures, stay on these pages or open the downloaded
notebook without executing cells. To redraw a saved analysis without fitting
UMAP or domains, load its labels and coordinates as described in
:doc:`../quickstart`; T6 also includes a final reopen cell. ``provided`` mode
performs analysis and is distinct from reopening saved coordinates.

Example size and analysis
-------------------------

These are prepared training node counts, not raw peak counts or resource
requirements. Profile epochs describe the selected examples. Runtime and
memory depend on the device, graph and analysis settings.

.. list-table:: Prepared examples and profile settings
   :header-rows: 1
   :widths: 12 15 18 15 15 25

   * - Tutorial
     - Spots
     - Peaks
     - Motifs
     - Epochs
     - Domain analysis
   * - T1
     - 2,187
     - 25,422
     - 839
     - 300
     - Target K13
   * - T2
     - 2,275
     - 31,284
     - 844
     - 300
     - Target K17
   * - T3
     - 9,215
     - 110,386
     - 852
     - 500
     - Target K13
   * - T4
     - 9,370
     - 84,998
     - 852
     - 500
     - Target K11
   * - T5
     - 2,488
     - 25,186
     - 836
     - 500
     - Target K3
   * - T6
     - 7,118
     - 120,380
     - 0
     - 500
     - Exact-K target16

T1–T5 use one configurable ``Workflow`` with exact-K domain search. ``DATASET``
is a run label; profiles supply editable defaults. Set ``ATAC_PATH``, optional
``MOTIF_PATH``, ``OUTPUT_DIR`` and ``TARGET_K`` for your own data.

T6 uses ``MultiSliceConfig`` with the sole ``cs_matched03`` profile.
The notebook defaults to train; optional provided mode reads selected
embeddings. Both call ``sr.analyze`` for exact-K target16 with no fixed
resolution. Sections 1–16 cover the complete input, training, analysis,
plotting, export and reopening workflow. The current saved outputs are under
``result/Embryo_multislice/cs_matched03/``, with input and execution identities
in ``tutorial_execution.json``. See :doc:`../multislice` for the model,
coordinate limits and saved-result loading.

T6 reference annotations appear immediately after reading counts. All 7,118
spots remain, including 81 Unknowns. Its spot and joint UMAP rows show
``Sections``, ``Spatial domains`` and ``Reference annotations`` on one saved
coordinate table per row. Unknown annotations and joint shared peaks are
gray. Reference labels and display transforms do not enter model training.

Notebook pages
---------------

.. toctree::
   :maxdepth: 1

   T1_Mouse_Embryo_E13_Spatial_ATAC_RNA
   T2_Mouse_Embryo_E13_Spatial_ATAC
   T3_Mouse_Brain_P22_Spatial_ATAC_RNA
   T4_Mouse_Brain_P22_Spatial_CUTTag_RNA_H3K27ac
   T5_Human_Tonsil_Spatial_ATAC
   T6_Mouse_Brain_Development_MISAR_seq
