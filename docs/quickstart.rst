Quick start
===========

Start with :doc:`T1 <tutorials/T1_Mouse_Embryo_E13_Spatial_ATAC_RNA>` to analyze
the included E13 mouse embryo embeddings. Reading the web tutorial shows its
saved outputs. Running its cells in provided mode computes the selected
downstream analyses again, including domains, UMAPs and feature scores.

Choose an environment
---------------------

.. list-table:: Installation paths
   :header-rows: 1
   :widths: 22 42 36

   * - Purpose
     - What you need
     - Instructions
   * - Read the tutorials
     - A browser; the pages include saved figures and code.
     - :doc:`tutorials/index`
   * - Run a tutorial
     - A complete checkout and the pinned Python 3.10.22 tutorial environment
       on Linux x86_64, or Linux under WSL2.
     - :doc:`installation`
   * - Use Python or CLI with your data
     - The package and analysis dependencies in a Linux environment. Package
       distributions omit tutorials, data and results.
     - :doc:`Package-only installation <installation>` and :doc:`continuous_pipeline`

Follow the pinned installation recipe, including its explicit CPU or CUDA
PyTorch selection, before opening T1. Provided mode needs the analysis stack
but no GPU, tmux, R scanner or complete checkpoint.

Open T1 from the checkout
-------------------------

From the repository root, with the tutorial environment active:

.. code-block:: bash

   python tools/prepare_data.py --dataset ME13_joint_2187
   jupyter lab tutorials/T1_Mouse_Embryo_E13_Spatial_ATAC_RNA.ipynb

Select the environment's Jupyter kernel. T1's first cell finds ``ROOT`` by
walking from the kernel's current working directory through its parents,
looking for both ``seareg/__init__.py`` and ``tutorials/``. Launching from the
checkout makes these paths discoverable. ``ROOT`` is the checkout directory,
not a directory that you need to create separately.

In that first cell, set ``RUN_MODE="provided"`` and check these four values:

.. list-table:: T1 configuration
   :header-rows: 1
   :widths: 24 76

   * - Setting
     - Source and meaning
   * - ``RUN_MODE``
     - Set to ``"provided"`` for existing embeddings. The source notebook
       defaults to ``"train"``.
   * - ``ATAC_PATH``
     - Resolved relative to ``CONFIG_PATH.parent`` from ``profile["input"]["atac"]``.
       The T1 profile is ``tutorials/configs/ME13_joint_2187.json`` and selects
       ``ROOT / "data/ME13_joint_2187/counts.h5ad"``.
   * - ``EMBEDDING_DIR``
     - ``ROOT / "result/ME13_joint_2187/training/embeddings/300_epoch"``;
       the included directory contains its own ``export.json`` and named TSVs.
   * - ``OUTPUT_DIR``
     - ``ROOT / "result" / DATASET`` by default. Choose a writable base;
       each execution reserves a new ``runs/<mode>_<time>_<id>/`` below it.

Run the cells in order from the edited first cell. Keep the bundled counts
paired with the selected export. The loader checks receipt hashes, IDs,
shapes and analyzability; it does not silently choose another result.
Outputs include analysis tables, enabled domain/UMAP panels, T1's feature
score maps, PNG/PDF/SVG figures and ``tutorial_execution.json``. See
:doc:`workflow` for paths and stage options.

For training, retain ``RUN_MODE="train"`` and follow the device, motif and
tmux setup in :doc:`installation`. It analyzes that new completed run's
exports.

Choose the loader for your assets
---------------------------------

.. list-table:: Three different result assets
   :header-rows: 1
   :widths: 22 40 38

   * - Asset
     - Contents
     - Appropriate use
   * - Complete ``TrainingRun``
     - A verified native training artifact, status and named exports;
       exact recovery additionally needs its compatible full-state checkpoint.
     - ``sr.load_run(training_path)`` and ``training.load_embeddings()``.
   * - Provided export directory
     - ``export.json`` plus named propagated spot/peak and optional motif TSVs;
       complete checkpoints are not bundled.
     - The tutorial's ``load_tutorial_embeddings`` with matching counts;
       downstream analysis, not exact training recovery.
   * - Saved analysis artifact
     - Its artifact manifest, ``analysis_manifest.json`` and the numerical
       outputs that were enabled when it was computed.
     - ``seareg.analysis.load_result``; read saved labels/coordinates
       without fitting domains or UMAP again.

For example, in the configured T1 notebook, its existing helper loads the
lightweight export directly:

.. code-block:: python

   embeddings = load_tutorial_embeddings(EMBEDDING_DIR, counts_path=ATAC_PATH)

For an analysis you already computed, select the actual analysis path from
its execution receipt. With the analysis runtime installed:

.. code-block:: python

   from seareg.analysis import load_result

   result = load_result("/absolute/path/to/saved/analysis")
   domains = result.domains()          # Requires saved domain output.
   spot_umap = result.spot_embedding() # Requires saved spot UMAP output.

See :doc:`api` for complete run loading and further result methods.

Move to your own inputs
-----------------------

Prepare a spot-by-peak H5AD with finite nonnegative counts, unique text spot
and peak IDs, and finite native coordinates in ``obsm["spatial"]``. Motif
matrix rows must match the retained genomic intervals in the same assembly
and coordinate convention; motif IDs name features, not tissue labels.
See :doc:`input_data` for filtering and alignment checks and
:doc:`motif_annotation` for existing matrices or scanning. Co-profiled RNA is
not automatically included in these tutorial models.

Use :doc:`T6 <tutorials/T6_Mouse_Brain_Development_MISAR_seq>` and
:doc:`multislice` for joint sections: it requires section IDs and shared peak
identities, and uses adaptive depth gating with within-section spatial
graphs and no motifs. Inspect unit four-side coordinate coverage before
training. Train/provided both perform native analysis. To read the current
saved labels, graph and UMAPs without fitting, reopen
``result/Embryo_multislice/adaptive_depth_gate/analysis`` with
``seareg.analysis.load_result``. Reference annotations are display inputs;
Unknown spots are retained. Original studies,
accessions and the included files are listed in :doc:`data_access`.
