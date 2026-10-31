Tutorial workflow
=================

The five single-section tutorials share a configurable interface for bundled examples and
user data. Install the pinned environment, edit the first code cell and run
the cells in order. Each JSON profile supplies defaults; the selected paths
and options determine the actual run.

T6 uses the dedicated multislice API with the same ``TrainingJob`` and
``TrainingRun`` contracts. Its sole ``cs_matched03`` profile exposes counts,
annotations, section shards, coverage preflight and exact-K analysis. It keeps
``RUN_MODE="train"`` by default; optional provided mode reads embeddings and
also runs ``sr.analyze``. Sections 1–16 use that selected source and save
``tutorial_execution.json`` with the actual training and analysis provenance.
The current saved T6 exports, analysis and figures are under
``result/Embryo_multislice/cs_matched03/``. Reopening its ``analysis/`` reads
the saved labels and coordinates without refitting. See :doc:`multislice`
for the formula, coordinate limits and result loading. The durable contracts
below also apply to T6.

Unified configuration
---------------------

Set ``ATAC_PATH``, optional ``MOTIF_PATH``, ``OUTPUT_DIR`` and ``TARGET_K`` for
your data. ``DATASET`` names the run and does not select a registered example.
Input and output paths may be outside the repository. ``FILTERING_OPTIONS``,
``NETWORK_OPTIONS``, ``TRAINING_OPTIONS``, ``ANALYSIS_OPTIONS`` and
``EXECUTION_OPTIONS`` expose the stage parameters. ``MOTIF_OPTIONS`` controls
reading, scanning or disabling motif associations. See the
`tutorial configuration guide <https://github.com/xmyjy/SEAREG/blob/main/tutorials/README.md#configure-the-inputs-and-parameters>`_.

An existing input file is used directly. Only a missing path that identifies
an included compressed file triggers checked archive extraction. Missing
counts are reported without selecting another dataset; missing motifs follow
the configured read/scan policy. Matrix, ID and
numerical checks still apply during the corresponding stages.

New model (all tutorials default)
----------------------------------------

For T1–T5, ``RUN_MODE="train"`` configures one ``sr.Workflow`` from the selected inputs
and options. It calls ``flow.filter()``, ``flow.prepare_motif()``,
``flow.prepare_network()`` and ``flow.prepare_training()``, then submits an
independent training job. Motif files may be unpacked during input preparation;
matrix import or R scanning occurs after filtering. Disabled motifs return
``None`` and the network contains the remaining node types.
The job uses tmux, with its command, request, method manifest, stdout/stderr,
epoch diagnostics, recovery script and ``TRAINING_HANDOFF.md`` on disk.
The notebook waits for this job; it does not own the training process.

.. code-block:: python

   job = sr.start_training_job(training_inputs, training_config,
                               RUN_DIR / "training", execution=execution,
                               name=DATASET)
   training = job.wait()
   from seareg_tutorial_embeddings import load_tutorial_embeddings
   embeddings = load_tutorial_embeddings(training, counts_path=ATAC_PATH)

Step 9 verifies and reads the actual completed run's final node exports.
Provided mode uses the same loader with ``EMBEDDING_DIR`` as its source.
Reconnect to an existing task with
``sr.load_training_job(job_path).wait()``. Recover an interrupted task through
its generated ``resume.sh``; a new attempt preserves the old files.

T6 follows the same stage order through native multislice APIs: prepare and
validate inputs, reserve a unique run, inspect backed counts and native XY,
then immediately display existing annotations. It calls
``sr.filter_multislice_counts``, records the motif-free stage, calls
``sr.build_multislice_network`` to persist the actual spot–peak and
within-section spatial graphs, and passes that network to
``sr.prepare_multislice_training``. The sampler consumes these saved graph
values. ``sr.start_multislice_training_job`` returns the same native job/run
handles, and the notebook loads only that completed run's own embeddings.
The raw-count ``prepare_multislice_training`` API remains compatible.
Provided mode launches no training.

``checkpoints/latest.pt`` is a complete recovery checkpoint containing model,
Adam, RNG and sampler state. It is atomically updated every 10 epochs and at
each result export. Retained numbered checkpoints
(``checkpoints/epoch_NNNN.pt``), named embeddings (``embeddings/<epoch>_epoch/``)
and evaluations start at epoch 300, repeat every 100 epochs, and are also
saved at the final epoch. The
save settings ``export_start_epoch=300`` and ``export_every=100`` are
independent of the scientific optimizer trajectory. A 300-epoch tutorial saves
its result at 300; a 500-epoch tutorial saves at 300, 400 and 500. Shorter runs
still save their final result. Recovery rejects incompatible implementations
or numerical environments. All tutorial profiles use this result-save policy
and rolling state every 10 epochs. T6's gate caches refresh every 50 epochs;
this is independent of result export. Its native checkpoints also restore all
saved gate, geometry, protection, CS and microgain caches without recomputing
them. Scientific input, mode,
source and numerical identities must match the saved checkpoint.

Output directories
-------------------

Both modes prepare the selected counts before reserving a unique
``OUTPUT_DIR/runs/<mode>_<time>_<id>/`` directory; provided mode also validates
the selected exports first. The default output base is ``result/<dataset>/``.
For T1–T5 training, preparation artifacts are indexed in that run's
``workflow.json`` and stored under ``artifacts/<stage>/<key>/``. T6 stores
immutable filtered counts, network and training inputs in ``filtered/``,
``network/`` and ``training_inputs/``. Training control files live in
``training_job/``, beside the initial ``training/`` output directory.

New models and exports stay with their run. T1–T5 bundled exports are under
``result/<example>/training/embeddings/<epoch>_epoch/``. T6's current final
exports are at ``result/Embryo_multislice/cs_matched03/500_epoch/``; the parent
result directory contains its saved analysis, figures and execution receipt. A
recovered attempt can have a different training path; the returned ``TrainingRun`` and the
execution receipt identify the completed attempt. To reanalyze an existing
result, select its actual ``embeddings/<epoch>_epoch/`` directory.

T1–T5 profiles default to physical CUDA card 0; the T6 example selects card 1.
Check the selected physical GPU before launch. Other users may choose any
available card; the example does not impose a global GPU0 restriction. Override it with
``SEAREG_PHYSICAL_GPU``. For CPU training, set
``TRAINING_OPTIONS["device"] = "cpu"`` and
``EXECUTION_OPTIONS["physical_gpu"] = None`` in the first cell. Provided mode runs independently of
these GPU settings. See :doc:`data_access` and :doc:`motif_annotation` for inputs.

Existing embeddings
-------------------

Optional ``RUN_MODE="provided"`` reads the explicit
``EMBEDDING_DIR`` and uses ``ATAC_PATH`` for spot IDs and native XY. The
directory's own ``export.json`` describes its named propagated spot/peak
matrices and optional motif matrix. The common loader validates the selected
exports and returns an ``EmbeddingSet``.
Training preparation and model training are skipped. No GPU,
motif file, R, complete model or tmux installation is required. Missing or
invalid exports raise a source error. The execution record identifies the
selected export. T6 records ``training_executed=false`` and
``training_run=null`` in provided mode, while retaining its configured
``training`` and ``execution`` dictionaries. Reading an export does not repeat
the training that created its bytes.

To read T6's current saved analysis without fitting, use
``seareg.analysis.load_result("result/Embryo_multislice/cs_matched03/analysis")``
and its ``domains()``, ``spot_embedding()`` and ``joint_embedding()`` methods.
The notebook's final section also reopens saved results. Provided node
exports support analysis; exact training recovery requires the compatible
native full-state checkpoint from the complete training run.

Downstream analysis
-------------------

Both paths feed their chosen ``embeddings`` into the same analysis cells.
All profiles use float64 row-L2, PCA20 and Scanpy 15-neighbor graphs. T1–T5 use
exact-K Louvain search, with example K13/17/13/11/3. Set ``TARGET_K`` for your
single-section data. T6 train/provided uses existing ``sr.analyze`` exact-K
search with target16 and no fixed resolution, PCA20 ``covariance_eigh`` and
Louvain seed0. Failure to reach K16 is reported. The saved analysis retains
the actual search trace and native domain labels from that execution.
Spot UMAP uses that graph. Joint UMAP fits all available normalized node types
directly. Both use seed 1234 and
``min_dist=0.08``. Use ``ANALYSIS_OPTIONS["domains"]``,
``ANALYSIS_OPTIONS["spot_umap"]`` and ``ANALYSIS_OPTIONS["joint"]`` to
select which results to compute, read and plot. Domain labels are added to
the requested UMAP plots when domain identification is enabled. Otherwise
those plots use the available node types without domain labels. Disabling
all three skips domain and UMAP outputs. T1's feature-score cell and T6's
reference-map cell are separate analyses/displays.
T1–T5 spatial plots retain native increasing-y coordinates. T6 uses reference
section orientations and circular markers; its display transforms affect copies,
while saved spatial coordinates and numerical results remain native.

Named node exports also support further feature-based analyses. See
:doc:`unified_framework` for the manuscript's cosine-profile definitions and
the downstream evidence used to interpret candidate regulatory relationships.

New figures are PNG plus editable vector PDF/SVG without horizontal grid
lines through the data area. T6's spot and joint UMAP rows show ``Sections``,
``Spatial domains`` and ``Reference annotations`` on one saved coordinate
table per row, keeping all shared peaks gray. Reopening/recoloring those
saved coordinates does not refit the analysis.
T6 records its domain display mapping and palette source beside the execution
receipt. The :doc:`multislice` guide explains the selected palette and how to
configure display frames for new data.

Execution records
------------------

``tutorial_execution.json`` records mode, inputs, source hashes, embedding
origin, analysis and stage times. T6's provided-mode receipt retains the
configured ``training`` and ``execution`` dictionaries and records
``training_executed=false`` and ``training_run=null``. A training job records
runner/worker PIDs, GPU, session, diagnostics,
checkpoint and exit status. Completion requires the target epoch, readable
final checkpoint, final exports and a verified completed state.

``analysis/spot_graph_metadata.json`` and the analysis manifest record the
neighbor algorithm and parameters under ``neighbor_execution``. Environment
records include the loaded and installed numerical-library versions and paths.

Interrupted completion and intermediate exports
-----------------------------------------------

If training completed but final job status recording was interrupted,
recovery verifies and accepts the completed output. If the final checkpoint
reached the target but output publication is incomplete, recovery selects
the latest available pre-target checkpoint. It reports an error when none
exists and retains all files; the target epoch is not increased.

``export.json`` records ``analysis_ready`` separately for propagated and ego
embeddings, plus per-entry zero-direction row counts. An intermediate warmup
export can be retained for inspection while marked ineligible for analysis.
The tutorial loader selects propagated embeddings and reports an ineligible
export explicitly. Final propagated exports must be finite and nonzero.
