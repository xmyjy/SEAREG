Package architecture
====================

SEAREG |release| keeps its implementation and public entry points in one
top-level ``seareg/`` package. Model equations, optimization, data preparation,
and analysis have explicit module responsibilities. ``Workflow`` coordinates
these components, and the same stages are available as individual functions.
The canonical stage contracts and v2 configuration are described in
:doc:`continuous_pipeline`.

Module responsibilities
-----------------------

.. list-table:: Implementation in the single package
   :header-rows: 1
   :widths: 27 73

   * - Module
     - Responsibility
   * - ``model.py``
     - The actual LightGCN propagation and loss definitions. ``SEAREG`` is an
       alias for ``LightGCN``.
   * - ``train.py``
     - Training setup and the optimization loop. ``checkpoint.py``,
       ``exports.py`` own checkpoint recovery and node exports.
       The public ``train_SEAREG``
       function launches training in an isolated worker.
   * - ``multislice/``
     - The sole cs_matched03 joint-section trainer. Internal original gate and
       propagation, eight-sector geometry, risk protection, donor coherence
       and four-side matched microgain own detached refresh caches. Balanced
       updates/global negatives reuse native jobs and exports; checkpoints
       preserve all caches, Adam, RNG and sampling state.
   * - ``jobs.py``
     - Detached tmux training jobs, progress records, reconnection and recovery.
   * - ``data.py``
     - Typed ``PreparedData``, ``EmbeddingSet``, ``InputArtifact``,
       ``PreparedNetwork``, and ``TrainingRun`` objects, with loading of
       completed training results.
   * - ``preprocessing.py``
     - Input inspection and the raw-count filtering stage, ``filter_counts``.
   * - ``graph.py``
     - Motif alignment, model-node compilation, actual spatial/heterogeneous
       graph persistence through ``build_network`` and network inspection.
   * - ``artifacts.py``
     - Immutable stage handles, relative dependency verification and static
       graph IO.
   * - ``annotation.py`` and ``edges.py``
     - Complete FIMO sites/relations and explicit edge selection, respectively.
   * - ``compat.py``
     - Input/configuration adapters and supported storage formats.
   * - ``sampling.py``
     - Fixed training-input preparation through ``prepare_training``, plus
       graph-edge and negative sampling used during training.
   * - ``metrics.py``
     - Training ranking and retrieval metrics.
   * - ``domains.py`` and ``embedding.py``
     - Spatial domain identification, spot UMAP, and joint spot–peak–motif UMAP,
       including standalone result loading.
   * - ``analysis.py``
     - Orchestration and loading of a combined analysis.
   * - ``plotting.py``
     - Visualization of spatial domains and saved UMAP coordinates.
   * - ``pipeline.py``
     - ``Workflow`` configuration, stage ordering, and stage reuse checks.
   * - ``config.py``
     - Validated preprocessing, training, and analysis configuration objects.
   * - ``utils.py`` and ``_analysis_io.py``
     - Artifact identities and shared persistence helpers.
   * - ``sequence.py``
     - Frozen peak-sequence extraction and pooled sequence background.
   * - ``_runtime.py``
     - Subprocess launch and worker execution, including device and thread
       settings before scientific runtime imports.
   * - ``__init__.py``, ``__main__.py``, and ``cli.py``
     - Lazy Python exports and command-line entry points.
   * - ``results.py``
     - Compatibility reexports of typed result objects.

Artifact schema v2 stores the stage identities and dependencies. Checkpoint
continuation checks training implementation, inputs, configuration and
numerical runtime.

Public entry points
-------------------

Use the package root in application code:

.. code-block:: python

   import seareg as sr

   flow = sr.Workflow("tutorials/configs/ME13_joint_2187.json")
   report = flow.check()

For individual stages, use ``sr.filter_counts``, ``sr.build_network``,
``sr.prepare_training`` and ``sr.train_SEAREG``. Motif preparation uses
``sr.prepare_sequences``, ``sr.annotate_motifs`` and ``sr.select_motif_edges``
for an explicit FIMO workflow, or ``sr.import_motif_edges`` for an existing
named matrix. See :doc:`continuous_pipeline` for stage inputs and examples.
``FilterConfig`` and ``NetworkConfig`` configure filtering and graph construction
separately; ``TrainConfig`` configures optimization. These classes are exported
at the package root.

``sr.analyze`` provides combined downstream analysis. ``sr.identify_domains``,
``sr.fit_spot_umap``, and ``sr.fit_joint_umap`` provide independent analyses.
Their ``AnalysisConfig`` is also exported at the root. The :doc:`api` describes
arguments and result types; :doc:`workflow` shows the complete sequence.

``sr.train_SEAREG`` accepts the fixed ``TrainingInputs`` returned by
``sr.prepare_training`` and returns a completed ``TrainingRun``. The tutorials
use ``sr.start_training_job`` for detached tmux training and
``sr.load_training_job`` to reconnect. ``Workflow`` is also available from
``seareg.pipeline``.

Root exports are resolved lazily. ``sr.SEAREG`` loads the model and Torch;
training and analysis load the runtime needed for their stages. See
:doc:`workflow` for the tutorial execution sequence, current-run step-9 embeddings
and analysis/display settings.

The :doc:`cli` exposes the single-section workflow stages. It requires either
``--config`` or ``--atac``; direct-input overrides cannot be mixed with JSON.
The dedicated multislice functions are Python APIs, not CLI stages. Exact
implementation links on :doc:`api` use this documentation build's Git commit.

Compatibility entry points
---------------------------

``sr.prepare_inputs`` remains supported as an input-identity helper: it records
an ``InputArtifact`` without preprocessing the counts. This differs from
``Workflow.prepare_inputs()``, an alias of ``Workflow.filter()`` returning
``FilteredCounts``. ``sr.prepare_network``
accepts that artifact and a ``PreprocessConfig`` through the compatibility
adapter, which uses the current filtering and network implementations. For new
code with explicit stages, use ``filter_counts`` and ``build_network`` above.

``train_SEAREG(network, config, output, *, execution=None, resume_from=None)``
accepts ``TrainingInputs`` or a prepared network. Import it through
``seareg.train_SEAREG`` or ``seareg.train.train_SEAREG``. The completed-run
loader is available as ``seareg.load_run`` or ``seareg.data.load_run``.

Existing results and new runs
------------------------------

The saved-result loaders read completed artifacts without training,
clustering, or fitting UMAP:

.. code-block:: python

   from seareg import load_run
   from seareg.analysis import load_result

   training = load_run("result/example/training")
   embeddings = training.load_embeddings()
   analysis = load_result("result/example/analysis")

These paths illustrate an existing completed run; replace them with its actual
location. Keep the complete run directory and its referenced inputs available
for the loaders' integrity checks.

Direct stage calls require an unused output directory. The ``Workflow``
interface reuses matching completed stages and branches when inputs or
configuration change; see :doc:`continuous_pipeline` for its layout.

Checkpoint compatibility includes executable AST identity, input fingerprints,
numerical configuration and runtime. Comments, whitespace and docstrings do
not change executable identity. ``export_start_epoch``, ``export_every`` and ``checkpoint_every`` are
output-management settings excluded from the scientific trajectory.
The shared default saves rolling recovery state every 10 epochs and full
checkpoints and embeddings from epoch 300 every 100 epochs, including
completion. A 500-epoch tutorial exports results at 300, 400 and 500. T6's
50-epoch gate-cache refresh is a separate model operation.
Upstream stage identities include source hashes, so rebuilding an input stage
after a source edit can produce a different training-input identity.
