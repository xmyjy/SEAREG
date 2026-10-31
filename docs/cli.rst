Command-line interface
======================

The installed ``seareg`` command (also ``python -m seareg``) exposes the
single-section :py:class:`seareg.Workflow`. The parser is in
:source:`seareg/cli.py`; Python stage contracts and output layouts are in
:doc:`continuous_pipeline`. Multislice training uses the dedicated Python
functions in :doc:`api` and :doc:`multislice`.

Choose one input mode
---------------------

Use a finalized JSON configuration for explicit paths, filtering, motif
selection, training, execution and optional analysis settings:

.. code-block:: bash

   seareg --config finalized.json check
   seareg --config finalized.json plan

Or supply a spot-by-peak H5AD directly. This example disables motif resources:

.. code-block:: bash

   seareg --atac counts.h5ad --output result/my_dataset --motif false check
   seareg --atac counts.h5ad --output result/my_dataset --motif false plan

``--config`` and ``--atac`` are mutually exclusive and one is required.
In configuration mode, put all input/training/execution settings in JSON:
combining ``--config`` with any direct-input option below is rejected rather
than overriding the file. ``--resume-from`` and ``--training-run`` remain
stage-specific selectors in either mode. Paths inside JSON resolve relative
to the configuration file; direct paths resolve from the current directory.
See :doc:`input_data` for counts, coordinate and identity requirements.

Stages and side effects
-----------------------

.. list-table:: Accepted positional stages
   :header-rows: 1
   :widths: 37 63

   * - Stage
     - Behavior
   * - ``check``, ``plan``
     - Inspect configuration/dependencies or show reusable/pending stages.
       No training, scanner, fitting or output-directory creation.
   * - ``filter``, ``prepare-inputs``
     - Save/reopen filtered counts. ``prepare-inputs`` aliases ``filter``.
   * - ``prepare-motif``
     - Import or generate selected motif edges; skipped for ``motif=false``.
   * - ``prepare-sequences``, ``annotate``, ``select-edges``
     - Explicit configured sequence/FIMO stages. The legacy R route has its
       own sequence extraction; these stages are not interchangeable scanners.
   * - ``prepare-network``, ``prepare-training``
     - Compile graphs, then fix splits and initial training shards.
   * - ``train``, ``run``
     - Synchronous training; ``run`` includes required upstream preparation
       and explicitly configured analysis/rendering.
   * - ``analyze``
     - Fit configured domains/UMAP from completed training; does not start
       missing training. ``--training-run PATH`` selects a completed run.
   * - ``render``
     - Draw saved analysis labels/coordinates without fitting.

``--resume-from CHECKPOINT`` is valid only for ``train``/``run`` and requests
explicit compatible checkpoint continuation. ``--training-run DIRECTORY`` is
valid only for ``analyze``. Neither an embedding export nor a completed run
directory is a substitute for a full resume checkpoint; see :doc:`workflow`.

Training commands require the scientific runtime, validated inputs, storage
for outputs and reviewed scientific settings. CUDA requires both
``--device cuda:0`` and ``--physical-gpu N``. The command waits for its worker;
launch long runs inside an independent tmux session or use
:py:func:`seareg.start_training_job` after preparing inputs. The notebooks use
this detached-job route. Run ``check`` and ``plan`` before executing:

.. code-block:: bash

   # Inside an independent tmux session, after reviewing settings and plan:
   seareg --config finalized.json run

Direct-input options
--------------------

All options in this table apply only with ``--atac``. Options omitted on the
command line retain the workflow/configuration defaults described in
:doc:`continuous_pipeline`; this table does not prescribe an experiment.

.. list-table:: Options accepted by the parser
   :header-rows: 1
   :widths: 39 61

   * - Option
     - Meaning
   * - ``--output PATH``
     - Workflow output base; omitted output is derived by the workflow.
   * - ``--motif {true,false}``
     - Enable motif inputs (default true). Also accepts ``yes/no`` and ``1/0``.
   * - ``--motif-source {auto,read,scan}``
     - Default ``auto`` reuses/builds a matrix; ``read`` requires an existing
       matrix; ``scan`` forces the original R scanner.
   * - ``--genome BUILD``, ``--fasta PATH``, ``--dna PATH``
     - Declared assembly, matching reference FASTA and named peak–motif matrix.
       Assembly can instead come from scalar count metadata.
   * - ``--rscript EXECUTABLE``
     - R command (default ``Rscript``); scanner dependencies must be installed.
   * - ``--no-auto-build-dna``, ``--no-download-reference``
     - Disable missing-matrix preparation or reference download, respectively.
   * - ``--epochs INT``, ``--hidden-dim INT``, ``--lr FLOAT``, ``--seed INT``
     - Explicit training options. Omitted epochs use filtered network size.
   * - ``--shards INT``, ``--pn-r INT``, ``--tr FLOAT``
     - Shards, negative ratio and training-positive fraction in ``(0, 1]``.
       Omitted shards use filtered positive-edge counts.
   * - ``--spatial-weight FLOAT``, ``--dna-weight FLOAT``
     - Explicit spatial and motif relation weights.
   * - ``--device DEVICE``, ``--physical-gpu INT``
     - Worker device (``cpu`` or ``cuda:0``) and selected physical CUDA card.

For motif-enabled input, supply its actual genome or metadata and either an
existing matrix or the R/reference resources. Scanning and downloads happen
at execution stages, not during ``check``/``plan``. See :doc:`motif_annotation`.
FIMO resource/selection options belong in JSON or the explicit Python APIs;
the separate ``seareg-annotate`` command is documented there.

The CLI prints JSON to stdout. A skipped motif stage reports ``state=skipped``;
completed stages report their path. Workflow errors print ``SEAREG: ...`` to
stderr and return exit code 1; argparse usage errors return 2.
