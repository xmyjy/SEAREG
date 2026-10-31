Python API
==========

Use ``import seareg as sr`` or import individual entry points from ``seareg``.
Root exports load their implementation modules on demand. See
:doc:`architecture` for module responsibilities.

This reference groups workflow, inputs/network, motifs, training/jobs,
multislice, results, analysis and plotting. Existing compatibility entry points
are retained below their relevant group. :doc:`continuous_pipeline` shows the
explicit stage sequence; :doc:`cli` is the command-line contract.

Implementation links use the exact Git revision of this documentation build:
:source:`workflow <seareg/pipeline.py>`,
:source:`inputs <seareg/preprocessing.py>`, :source:`network <seareg/graph.py>`,
:source:`motifs <seareg/annotation.py>`, :source:`edge selection <seareg/edges.py>`,
:source:`sampling <seareg/sampling.py>`, :source:`training <seareg/train.py>`,
:source:`jobs <seareg/jobs.py>`, :source:`multislice stages <seareg/multislice/stages.py>`,
:source:`multislice inputs <seareg/multislice/inputs.py>`,
:source:`multislice jobs <seareg/multislice/jobs.py>`,
:source:`results <seareg/data.py>`, :source:`defaults <seareg/defaults.py>`,
:source:`domains <seareg/domains.py>`, :source:`UMAP <seareg/embedding.py>`,
:source:`analysis <seareg/analysis.py>`, :source:`plotting <seareg/plotting.py>`.

Workflow
--------

.. py:function:: seareg.run_workflow(atac, *, output=None, motif=True, motif_source='auto', genome=None, fasta=None, dna=None, rscript='Rscript', auto_build_dna=True, download_reference=True, epochs=None, hidden_dim=None, lr=None, seed=None, device=None, shards=None, pn_r=None, tr=None, spatial_weight=None, dna_weight=None, physical_gpu=None, training=None, network=None, execution=None, analysis=None, resume_from=None)

   Run the counts-to-training workflow in one call using the same artifact DAG
   as the step-by-step API. Common non-None hyperparameters override matching
   fields of ``training`` (a TrainConfig or mapping). Omitted ``epochs`` and
   ``shards`` are resolved from the prepared filtered network; explicit fields
   take precedence. ``network`` and ``execution`` also accept configuration mappings.

   Return a completed ``TrainingRun`` with an output ``.path`` by default.
   Explicit ``analysis`` additionally requests analysis/rendering and returns
   ``AnalysisResult``. CUDA requires ``device='cuda:0'`` and an explicit
   ``physical_gpu``. Completed matching stages are reused, while ``resume_from``
   is an explicit checkpoint continuation request subject to existing guards.
   This call is synchronous. For independent long-running training, use the
   ``prepare_training`` / ``start_training_job`` route in :doc:`continuous_pipeline`.

.. py:class:: seareg.Workflow(config)

   Accept a JSON path (string/``Path``) or configuration dictionary and return
   a workflow coordinator. Relative paths resolve from the JSON file's parent
   directory, or the current directory for a dictionary. Construction does not
   start training; ``check()``/``plan()`` inspect it before executing stages.

   .. py:method:: from_counts(atac, *, motif=True, motif_source='auto', output=None, genome=None, fasta=None, dna=None, auto_build_dna=True, rscript='Rscript', download_reference=True, training=None, network=None, execution=None, analysis=None)

      Construct the same workflow without a JSON file. ``motif=False`` skips
      all motif resources. True reuses a matrix or generates a missing graph
      with the original R scanner; ``motif_source`` can force read/scan.
      A partial ``training`` mapping enables automatic omitted epochs/shards.
      A complete ``TrainConfig`` makes all of its values explicit.

   .. py:method:: prepare_motif()

      Return ``SelectedMotifEdges`` after optional import/scanning, or ``None``
      when ``motif=False``. This step does not train a model. Continue with
      ``prepare_network()``, ``prepare_training()`` and ``train()``.

   .. py:method:: check()

      Return a dictionary describing the inputs, dependencies and execution
      settings. Perform no training, fitting or output writes.

   .. py:method:: prepare_inputs()

      Compatibility alias of ``filter()``; return frozen ``FilteredCounts``.

   .. py:method:: plan(target='run')

      Show reusable, new or pending stages without
      creating directories, decoding counts, scanning or training.

   .. py:method:: filter()

      Save/reopen the unique filtered raw counts and all peak-filter decisions.

   .. py:method:: prepare_sequences()

      Save/reopen peak FASTA, axis mapping and pooled background. The legacy R
      route extracts its own FASTA and does not use this FIMO sequence stage.

   .. py:method:: annotate()

      Explicitly run or reopen configured local motif annotation and return an
      ``AnnotationResult``. Does not train. See :doc:`unified_framework`.

   .. py:method:: prepare_network()

      Return a ``ModelNetwork`` after preprocessing, or load an explicitly
      configured matching prepared network.

   .. py:method:: select_edges()

      Save/reopen explicit named selected motif edges and selection decisions.

   .. py:method:: prepare_training()

      Save/reopen fixed split, initial shards and full training-positive graph.

   .. py:method:: train(*, resume_from=None)

      Prepare the network if needed, train synchronously in a worker and return
      a completed ``TrainingRun``. Use ``start_training_job`` for a detached job.

   .. py:method:: analyze(training=None)

      Return an ``AnalysisResult`` from completed training. ``training`` can be
      a ``TrainingRun`` or its directory path. Omission uses this workflow's
      completed training stage.

      The default selects the last successful ``train()`` recorded by this
      workflow, including a resumed branch, after verifying its current
      inputs, configuration, implementation and artifact identity. Reopening
      the workflow keeps that choice; it does not need the old checkpoint
      file to analyze a completed resumed run. ``plan("analyze")`` uses the
      same selection. Explicit ``training=...`` overrides this call.

      CLI users can select a completed result directly with
      ``seareg --config config.json analyze --training-run /path/to/run``.

   .. py:method:: render(analysis=None)

      Render already saved labels and coordinates without fitting or training.
      With ``analysis`` omitted, resolve the analysis for this workflow's
      default completed training stage. To render an explicitly selected
      training result, pass the object returned by ``analyze(training=...)``:

      .. code-block:: python

         analysis = flow.analyze(training=chosen_training)
         flow.render(analysis)

   .. py:method:: run(*, resume_from=None)

      Automatically execute all required preparation and training stages and
      return ``TrainingRun``. If analysis is explicitly configured, continue
      to analysis/optional rendering and return ``AnalysisResult``.
      The call waits for its worker; it does not submit an independent job.

Continuous stage API
--------------------

These are the canonical inputs/network and motif stages. ``output`` accepts a
string or ``Path`` naming an unused directory. A handle or its directory can
be supplied wherever a saved artifact is accepted. Direct creation calls do
not overwrite/reuse a directory; use the handle's ``load`` method to reopen
verified completed work. ``Workflow`` selects matching stages automatically
from their input/configuration/code identities. Keep referenced upstreams
available. See :doc:`continuous_pipeline` for the filenames and
:doc:`T1 <tutorials/T1_Mouse_Embryo_E13_Spatial_ATAC_RNA>` for the tutorial flow.

.. py:function:: seareg.filter_counts(atac, config, output)

   Accept an H5AD path and ``FilterConfig``; return ``FilteredCounts``.
   ``output/counts.h5ad``, ``peak_filter.tsv.gz`` and ``filter_report.json``
   preserve the filtered raw counts and decisions. No graph/model is fitted.

.. py:function:: seareg.build_network(filtered, selected, config, output, *, _compiled_data=None)

   Accept ``FilteredCounts``, optional ``SelectedMotifEdges`` (``None`` for
   motif-free data) and ``NetworkConfig``; return ``ModelNetwork`` with static
   graph files, model axes and lineage under ``output`` (including
   ``data.json``, ``graph_report.json`` and configured spatial/motif adjacency
   files). ``_compiled_data`` is
   an internal compatibility argument; leave it at ``None`` in new code.

.. py:function:: seareg.prepare_training(network, config, output)

   Accept ``ModelNetwork`` and ``TrainConfig``; return ``TrainingInputs``.
   Save ``split.npz`` (fixed split/initial shards),
   ``full_train_adjacency.npz`` and ``sampling.json`` under ``output``.
   No optimization occurs. Reuse requires matching sampling
   settings as well as the upstream network.

.. py:function:: seareg.prepare_sequences(filtered, config, output)

   Accept ``FilteredCounts`` and ``SequenceConfig`` (local FASTA and declared
   assembly); return ``PeakSequences``. Save ``peaks.fa``, ``peaks.bed``,
   ``peak_sequences.tsv`` and ``background.json`` under ``output``.

.. py:function:: seareg.annotate_motifs(sequences, config, output, *, attempts_dir=None)

   Accept ``PeakSequences`` and ``AnnotationConfig`` with local MEME resource
   and FIMO executable; return ``MotifAnnotation`` (``AnnotationResult`` alias).
   Save complete ``motif_sites.tsv.gz``, ``peak_motif_relations.tsv.gz`` and
   named axes under ``output``. ``attempts_dir`` optionally selects attempt
   logs. See :doc:`motif_annotation` for scanner prerequisites.

.. py:function:: seareg.import_motif_sites(sequences, config, hits, output)

   Import a FIMO text-table path using the same sequences/configuration and
   return ``MotifAnnotation`` with the same saved contract. Consistency checks
   do not establish that the external scan was exhaustive.

.. py:function:: seareg.select_motif_edges(annotation, config, output)

   Accept ``MotifAnnotation`` and ``EdgeSelectionConfig``; return
   ``SelectedMotifEdges``. Save selected sparse edges and selection decisions
   under ``output``. ``all_candidates`` or ``site_p`` is an explicit scientific
   choice; this artifact is distinct from the complete annotation.

.. py:function:: seareg.import_motif_edges(source, output, *, upstream=None)

   Import a named peak–motif HDF5 matrix path as ``SelectedMotifEdges`` under
   ``output``. ``upstream`` optionally binds its generating annotation.
   This is the supplied-matrix route used by T1–T5.

The canonical functions are ``filter_counts``, ``prepare_sequences``,
``annotate_motifs``, ``import_motif_sites``, ``select_motif_edges``,
``import_motif_edges``, ``build_network`` and ``prepare_training``. They return
``FilteredCounts``, ``PeakSequences``, ``MotifAnnotation``,
``SelectedMotifEdges``, ``ModelNetwork`` and ``TrainingInputs`` handles.
Stage examples, saved filenames and v1/v2 configuration are in
:doc:`continuous_pipeline`.

Configuration objects
---------------------

.. code-block:: python

   from seareg import PreprocessConfig, TrainConfig, AnalysisConfig

   preprocessing = PreprocessConfig()
   training = TrainConfig(epochs=300, device="cpu", seed=1234)
   analysis = AnalysisConfig(target_domains=(13,))

``PreprocessConfig`` controls spatial graph construction, motif alignment and
the filtering protocol. ``TrainConfig`` controls optimization and graph
sampling. ``AnalysisConfig`` controls domain identification and UMAP. The
``target_domains`` tuple must be supplied when domain analysis is enabled,
or set ``fixed_resolution`` to fit once and retain the observed domain count.
These options are mutually exclusive. ``pca_solver`` defaults to ``auto``;
T6 explicitly uses ``covariance_eigh`` for its selected analysis protocol.
JSON configurations reject unknown keys and validate each option's type.

``sr.training_profile('balanced', **overrides)`` returns an explicit efficient
``TrainConfig``. ``speed`` and ``memory`` presets are also available. JSON uses
the optional root ``training_profile`` string; explicit ``training`` values
override the preset. See
:doc:`unified_framework` for loss workspace, residency and telemetry limits.

New-data training defaults
--------------------------

.. py:function:: seareg.training_defaults(n_spots, positive_edges)

   Return an epoch/shard dictionary from filtered binary graph statistics.
   ``n_spots`` is the total training spot count: use 500 epochs above 2,500
   spots, otherwise 300. An integer ``positive_edges`` gives
   ``shards = max(10, ceil(positive_edges / 700000))``. Count each retained
   nonzero spot–peak pair once before the train/test split or negative sampling,
   excluding motif and spatial edges.

   A mapping from actual section IDs to filtered edge counts instead returns
   ``slice_shards`` with that rule applied per section and
   ``updates_per_epoch = max(slice_shards.values())``. Shard counts are not
   rounded to multiples of ten.

.. py:function:: seareg.training_config(network, **overrides)

   Read validated filtered spot and positive-edge counts from a prepared
   ``ModelNetwork`` and return a ``TrainConfig`` with automatic epochs/shards.
   Explicit keyword overrides take precedence. For example:

   .. code-block:: python

      config = sr.training_config(network, device='cpu')
      inputs = sr.prepare_training(network, config, 'run/training_inputs')

.. py:function:: seareg.multislice_training_config(counts, **overrides)

   Read actual section IDs from AnnData or an H5AD path, apply the existing
   multislice peak filter and return a ``MultiSliceConfig``. Shards use each
   section's filtered positive edges; epochs use the joint total spot count.
   ``updates_per_epoch`` uses the largest final section shard count unless
   explicitly overridden. Resolve this configuration before creating immutable
   filtering, network and training-input artifacts.

   For very small inputs with too few positives to populate ten shards,
   pass smaller explicit ``slice_shards`` values.

Schema-v2 ``Workflow``, ``run_workflow`` and CLI entries fill only omitted
epoch/shard fields after network preparation. Explicit configuration fields
and helper keyword arguments take precedence. The existing ``TrainConfig``
and ``MultiSliceConfig`` defaults and supplied tutorial configurations retain
their values. See :doc:`continuous_pipeline` for complete examples.

The five single-section workflow profiles and multislice profile
``Embryo_multislice_adaptive_depth_gate.json`` are in ``tutorials/configs/``. See
:doc:`workflow` for their main parameters and :doc:`input_data` for preprocessing
behavior.

Multislice training
--------------------

The following functions accept ``MultiSliceConfig`` and unused output paths.
Use the same resolved configuration across preparation and submission. See
:doc:`T6 <tutorials/T6_Mouse_Brain_Development_MISAR_seq>` and :doc:`multislice`
for shared peak identities, native section coordinates and saved-run provenance.

.. py:function:: seareg.filter_multislice_counts(counts, config, output)

   Accept AnnData or an H5AD path with section IDs and native coordinates;
   return ``MultiSliceFilteredCounts``. Save ``counts.npz``, coordinates and
   ordered axes under ``output`` using the selected per-section/shared filter.

.. py:function:: seareg.build_multislice_network(filtered, config, output)

   Accept ``MultiSliceFilteredCounts`` or its path; return ``MultiSliceNetwork``
   with static spot–peak and within-section spatial adjacency files and graph
   report under ``output``. It references the matching filtered artifact.

.. py:function:: seareg.prepare_multislice_training(counts, config, output)

   Accept ``MultiSliceNetwork`` or its path and return ``MultiSliceInputs``
   bound to the saved graphs. AnnData/H5AD is also accepted by the original
   raw-count preparation route. Save counts/axes and preparation/sampling
   identities under ``output``; reference labels are not training inputs.

.. py:function:: seareg.start_multislice_training_job(inputs, config, output, *, execution=None, name=None)

   Accept ``MultiSliceInputs`` or its path; require tmux and matching config.
   Return ``TrainingJob`` immediately. Training writes to ``output`` and its
   sibling ``<output-name>_job`` stores logs, handoff and recovery script.
   Both directories must be unused. Reconnect with ``load_training_job``;
   ``job.wait()`` returns a completed native ``TrainingRun``.

.. code-block:: python

   import seareg as sr

   counts = "counts.h5ad"
   config = sr.multislice_training_config(
       counts, device="cpu", correction_mode="adaptive_depth_gate")
   filtered = sr.filter_multislice_counts(counts, config, "run/filtered")
   network = sr.build_multislice_network(filtered, config, "run/network")
   inputs = sr.prepare_multislice_training(network, config, "run/inputs")
   job = sr.start_multislice_training_job(
       inputs, config, "run/training", execution={"physical_gpu": None})
   training = job.wait()  # Native TrainingRun from this training job.
   embeddings = training.load_embeddings()

``MultiSliceInputs`` stores counts, peak/spot identities, section IDs and native
coordinates. Preparation applies the selected per-section feature thresholds
and shared-peak filtering. No reference tissue labels are accepted.
``correction_mode`` accepts only ``"adaptive_depth_gate"``. The internal
``AdaptiveDepthGate`` retains original ``g`` and donor weights ``1-g``;
reception is ``g*(1-P*rho)+a*C+e`` with ``e=alpha_s*H*V``. ``Q=O*D`` controls
a matched local-3% reference budget independently in each section. Full-message
caches are detached and refresh every 50 epochs after loss and before export.
See :doc:`multislice` for equations, numerical order and fallback conditions.
No new trainable gate parameters or target-domain head are added.

The T6 profile uses 500 epochs and explicit 10/10/20/20 shards. Its
train/provided analysis requests ``target_domains=[16]`` and
``fixed_resolution=None`` through ``sr.analyze``. This native analysis saves
the actual search trace, labels and UMAP coordinates.
The current example outputs and execution receipt are under
``result/Embryo_multislice/adaptive_depth_gate/``. Reopen the saved ``analysis/`` with
``seareg.analysis.load_result`` to read those results without fitting.

The multislice default saves rolling state every 10 epochs, full checkpoints
and embeddings from epoch 300 every 100 epochs plus completion, matching the
single-section policy. A 500-epoch tutorial exports at 300, 400 and 500.
The 50-epoch gate refresh remains separate from result saving and optimization.
Native recovery restores model and all gate,
geometry, protection, coherence and displacement-budget caches, Adam, RNG and sampler
state; input, formula, implementation and numerical identities must match.
Provided node exports support analysis; exact resume requires the compatible
full-state checkpoint and complete training run.

``seareg.multislice.geometry.spatial_coverage_report(coordinates, spatial,
slice_codes)`` reports ``total_spots``, ``unit_cardinal_distance=1``,
``cardinal_tolerance=1e-6`` and a ``per_slice`` list with ``slice_code``,
``spots``, ``eight_sector_covered``, ``eight_sector_fraction``,
``cardinal_covered`` and ``cardinal_fraction``. It inspects the original graph
and native coordinates, not display transforms. It does not infer alpha or
budgets before full-message refresh. See :doc:`input_data` for coordinate
requirements and :doc:`workflow` for durable jobs and recovery.

Inputs and networks
-------------------

The canonical stage functions are listed in `Continuous stage API`_ above.
The following input-identity and network adapters compose the preparation stages.
Root ``seareg.prepare_inputs`` returns ``InputArtifact`` without filtering;
``Workflow.prepare_inputs()`` instead aliases ``filter()`` and returns
``FilteredCounts``. Root ``prepare_network`` accepts the legacy input identity
and ``PreprocessConfig``; ``Workflow.prepare_network()`` coordinates canonical
stages and uses the workflow's ``NetworkConfig``.

.. py:function:: seareg.preprocessing.inspect_inputs(atac, dna=None, *, use_spatial=True)

   Return input structure, coordinate and file-identity information. Count
   values are validated when filtering; motif values are validated when
   importing and aligning motif edges.

.. py:function:: seareg.prepare_inputs(atac, output, dna=None, *, use_spatial=True)

   Create an input identity directory and return an ``InputArtifact``. The
   source count and annotation files remain at their supplied paths. Also
   available as ``seareg.preprocessing.prepare_inputs``.

.. py:function:: seareg.load_inputs(path)

   Load an ``InputArtifact`` and check that its referenced input files match.
   Its ``paths`` property resolves the recorded local file locations. Also
   available as ``seareg.data.load_inputs``.

.. py:function:: seareg.prepare_network(inputs, config, output)

   Build a ``ModelNetwork`` (also exported as ``PreparedNetwork``) from an
   ``InputArtifact`` or its directory path and a ``PreprocessConfig``. Also
   available as
   ``seareg.graph.prepare_network``.

.. py:function:: seareg.graph.load_network(path)

   Load a completed prepared network. ``network.load_data()`` returns the
   binary count matrix, aligned DNA associations, ordered IDs and coordinates.

.. py:function:: seareg.graph.inspect_network(path)

   Return the prepared network fingerprint, node counts and counts of positive
   spot–peak and peak–DNA edges.

Peak–motif generation
----------------------

Canonical motif stages are in `Continuous stage API`_. These compatibility
functions compose filtering, sequence preparation and annotation/export;
existing anchors remain supported. Source adapters are in
:source:`seareg/compat.py` and :source:`seareg/sequence.py`; the R scanner is
:source:`seareg/scanners/legacy.py`.

.. py:function:: seareg.annotate_peak_motifs(atac, fasta, assembly, motifs, output, *, preprocessing=None, fimo="fimo", p_cutoff=1e-5, motif_pseudocount=0.0, timeout=3600)

   Run local FIMO on the final filtered peaks with explicit DNA MEME matrices.
   Return an ``AnnotationResult`` preserving all axes and candidate sites,
   including zero-hit motifs. Select a training graph explicitly through
   ``export_motif_graph``. Input types match the legacy entry point below.

.. py:function:: seareg.import_fimo_annotation(atac, fasta, assembly, motifs, hits, output, *, preprocessing=None, p_cutoff=1e-5)

   Validate an existing FIMO text table against the local reference and axes.
   Return the same candidate-annotation artifact type. Validation establishes
   internal consistency, not exhaustiveness of an externally supplied scan.

.. py:function:: seareg.export_motif_graph(annotation, output, *, selection, p_cutoff=None)

   Export a separate sparse named training graph from an ``AnnotationResult``
   or its path. Explicit ``selection`` is ``all_candidates`` or ``site_p``;
   the latter requires a cutoff no larger than the scan threshold. See
   :doc:`unified_framework` for candidate-graph interpretation.

.. py:class:: seareg.AnnotationResult

   ``load(path)`` verifies the saved manifest. ``motif_ids`` preserves literal
   IDs; ``load_matches()`` reloads the complete sparse FIMO candidate matrix.
   Legacy annotations provide loading and named motif access; complete
   candidate-site matrices are available only for FIMO annotations.

.. py:function:: seareg.scan_peak_motifs_legacy(atac, fasta, assembly, output, *, preprocessing=None, species_names=("Homo sapiens", "Mus musculus"), p_cutoff=1e-5, threads=1, rscript="Rscript", timeout=3600, cache_dir=None)

   Generate a peak–motif training graph with the bundled
   JASPAR2024/motifmatchr R scanner. ``atac`` accepts a counts H5AD path, a
   ``PreparedData``, a ``PreparedNetwork``, or its directory. For H5AD input,
   ``preprocessing`` is a ``PreprocessConfig`` applied before sequence
   extraction; omit it for an already fixed prepared peak axis. Return the
   new annotation directory containing ``freq_motif.h5`` and provenance.
   See :doc:`motif_annotation` for R dependencies, effective pruning settings,
   output semantics and the ``seareg-annotate`` command.

.. py:function:: seareg.scan_annotation(prepared, fasta, assembly, command, output, scanner_args=(), timeout=3600)

   Run a caller-selected scanner on validated ``PreparedData``. ``command``
   is an explicit executable argument list, such as
   ``["Rscript", "/path/scanner.R"]``. The scanner accepts ``-i`` BED,
   ``-g`` FASTA and ``-o`` output arguments and writes named
   ``freq_motif.h5`` or ``freq_kmer.h5``. Validate its output peak order and
   matrix before publishing the directory. This adapter is also available as
   ``seareg.sequence.scan_annotation``.

Training and embeddings
-----------------------

.. py:function:: seareg.start_training_job(training_inputs, train_config, output, *, execution=None, name=None)

   Accept ``TrainingInputs`` or its directory and matching ``TrainConfig``.
   Require tmux and the scientific runtime; return ``TrainingJob`` immediately.
   Training outputs go under the unused ``output`` directory, while its unused
   sibling ``<output-name>_job`` stores logs, ``TRAINING_HANDOFF.md`` and
   ``resume.sh``. ``execution`` is a mapping of worker/device settings.
   ``job.wait()`` returns ``TrainingRun``. See :doc:`workflow` for progress,
   checkpoint compatibility and precise recovery.

.. py:function:: seareg.load_training_job(path)

   Reconnect to a saved job directory and return ``TrainingJob`` without
   resubmitting it. Inspect its handoff/progress; call ``wait()`` to wait for
   completion or ``resume(checkpoint=...)`` for explicit compatible recovery.
   This loader does not fit domains or UMAP.

.. py:class:: seareg.SEAREG(n_spots, n_peaks, n_dna, dim, n_layers=2)

   Alias for ``seareg.model.LightGCN``, the two-layer SEAREG model. Accessing
   this class imports Torch. Constructing it initializes model parameters;
   use ``train_SEAREG`` or ``Workflow.train`` for the complete training stage.

.. py:function:: seareg.train_SEAREG(network, config, output, *, execution=None, resume_from=None)

   Train from ``TrainingInputs``, a ``ModelNetwork`` or compatible prepared
   input using a ``TrainConfig``. Legacy inputs normalize once into the same
   immutable network and split stages.
   Execute training synchronously in a worker and return a completed ``TrainingRun``.
   For long runs, use ``start_training_job`` to submit an independent tmux job;
   see :doc:`workflow` for waiting, reconnecting and recovery.
   ``execution`` supplies the worker device/thread settings. An explicit
   ``resume_from`` must pass the trainer's compatibility checks, including
   fixed input identity, executable source structure and numerical runtime.
   Use a new run output directory. See :doc:`architecture` for checkpoint
   compatibility. Also available as
   ``seareg.train.train_SEAREG``; the model and optimization implementations
   reside in ``model.py`` and ``train.py``, respectively.

Results and saved embeddings
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. py:function:: seareg.load_run(path)

   Load a completed ``TrainingRun`` without continuing training, including a
   compatible saved result from an earlier package version. Also available
   as ``seareg.data.load_run``.

.. py:method:: seareg.data.TrainingRun.load_embeddings(epoch=None, representation="propagated")

   Load an ``EmbeddingSet`` with ordered spot, peak and DNA-feature identities.
   By default, use the final exported epoch and propagated representation.
   An explicit epoch must be available in the run. ``run.final_epoch`` reports
   the final exported epoch.

Combined analysis
-----------------

.. py:function:: seareg.analyze(inputs, config, output, *, plots=True)

   Compute an analysis from an ``EmbeddingSet`` or ``TrainingRun`` and an
   ``AnalysisConfig``. Save the requested domains and embeddings, with figures
   when ``plots=True``, and return an ``AnalysisResult``. ``output`` must be an
   unused string/``Path`` directory. Depending on enabled methods, it contains
   ``spatial_domains_K*.tsv`` or ``spatial_domains_fixed_resolution.tsv``,
   ``spot_umap.tsv``, ``joint_umap.tsv`` and ``analysis_manifest.json``.
   Reopen with ``seareg.analysis.load_result`` to reuse the saved fits.
   Also available as ``seareg.analysis.analyze``.

.. py:function:: seareg.analysis.load_result(path)

   Load a completed analysis without fitting. The returned object provides
   ``domains(k=None)``, ``spot_embedding()`` and ``joint_embedding()``.

``analysis.domains(k)`` returns a ``DomainResult`` with ``labels`` (a DataFrame
containing ``spot_id``, ``slice_id`` and ``domain``), ``coordinates``, ``graph``
and the resolution-search ``trace``. Omit ``k`` only when the saved analysis
contains exactly one domain partition.

``analysis.spot_embedding()`` and ``analysis.joint_embedding()`` return
``EmbeddingResult`` objects. Their ``table`` attributes hold the saved UMAP
coordinates. Joint tables preserve ``entity_type``, ``entity_id`` and
``native_row`` as well as ``UMAP1`` and ``UMAP2``.

Independent analyses
--------------------

Spatial feature scores
----------------------

.. py:function:: seareg.spatial_scores(inputs, *, peak_ids=(), motif_ids=(), standardize=False)

   Calculate requested spot–peak and spot–motif cosine associations from an
   ``EmbeddingSet`` or completed ``TrainingRun``. IDs must match the native
   peak/motif embedding indices. Only the requested feature profiles are formed.
   Also available as ``seareg.analysis.spatial_scores``.

   The long DataFrame contains ``spot_id``, ``slice_id``, ``feature_type``,
   ``feature_id``, ``cosine`` and ``score``. Native ``x``/``y`` columns are
   included when coordinates are available. Spot order and feature-request
   order are retained, and peak/motif identities remain distinct.

   Embeddings are normalized to unit length in float64. By default ``score``
   is the raw cosine. With ``standardize=True``, it is the per-feature z-score
   across all supplied spots, with population standard deviation (``ddof=0``).
   This gives the E13 peak and motif spatial association scores (PSAS/MSAS).
   Constant profiles cannot be standardized; their raw cosine remains available
   with ``standardize=False``. Motif requests require motif embeddings.

   This operation performs no training, domain/UMAP fitting, smoothing, clipping
   or file export. These association scores are not fragment counts or RNA
   expression values.

.. code-block:: python

   scores = sr.spatial_scores(
       embeddings,
       peak_ids=["chr4-22497217-22498484"],
       motif_ids=["PRDM1_MA0508_4"],
       standardize=True,
   )
   scores.to_csv("feature_spatial_scores.tsv", sep="\t", index=False)

The :doc:`T1 tutorial <tutorials/T1_Mouse_Embryo_E13_Spatial_ATAC_RNA>` shows
MSAS/PSAS on native spatial coordinates with square spots
and a shared viridis z-score scale, alongside discrete observed peak detection
(raw count > 0). It exports the scores, raw counts/detection values and editable
figures.

Domains and UMAPs
-----------------

.. py:function:: seareg.identify_domains(inputs, config, output=None)

   Compute one partition and return a ``DomainResult``. Supply exactly one
   ``config.target_domains`` value for an exact-K resolution search, or set
   ``config.fixed_resolution`` with an empty target tuple for one Louvain fit.
   These modes are mutually exclusive. ``observed_k`` reports the actual domain
   count; ``target_k`` is ``None`` in fixed-resolution mode. If ``output`` is
   provided, also save the standalone result. Also available as
   ``seareg.domains.identify``.

.. py:function:: seareg.fit_spot_umap(inputs, config, output=None)

   Cast spot embeddings to ``float64``, normalize each row to unit L2 norm,
   fit the configured PCA, build Scanpy neighbors from those components, and
   run Scanpy UMAP. Return an ``EmbeddingResult`` and save a standalone result
   when ``output`` is provided. The tutorial profiles use 20 PCs, 15 Euclidean
   neighbors, ``min_dist=0.08``, seed ``1234``, and two output dimensions.
   Also available as ``seareg.embedding.fit_spot``.

.. py:function:: seareg.fit_joint_umap(inputs, config, output=None)

   Cast each node type to ``float64``, normalize its rows to unit L2 norm,
   concatenate all spot, peak, and motif rows, and fit ``umap.UMAP`` directly.
   Return an ``EmbeddingResult`` and optionally save it. The joint projection
   uses every node, without PCA, sampling, or feature-to-spot alignment.
   The tutorial profiles use 15 Euclidean neighbors, ``min_dist=0.08``, seed
   ``1234``, and two output dimensions. Also available as
   ``seareg.embedding.fit_joint``.

Use ``seareg.domains.load_result(path)`` to read a standalone domain result and
``seareg.embedding.load_result(path)`` to read a standalone spot or joint UMAP.
These loaders do not fit new results.

Plotting
--------

Accept saved pandas coordinate tables or aligned NumPy arrays; see
:doc:`workflow` for reopening labels/coordinates without refitting. These
functions return ``(Figure, Axes)`` without saving; ``save_figure`` is the
explicit export operation. T1 and :doc:`T6
<tutorials/T6_Mouse_Brain_Development_MISAR_seq>` demonstrate saved panels.

.. py:function:: seareg.plotting.spatial_domains(coordinates, labels, title="", ax=None, point_size=20, annotate_domains=False, *, legend=True, palette=None, marker="o")

   Draw domains at native spatial coordinates and return ``(Figure, Axes)``.
   ``point_size`` is the scatter area in square points. Domain annotations
   are optional; the default plot uses a legend.

.. py:function:: seareg.plotting.spot_umap(table, labels=None, title="SEAREG (spots)", ax=None, point_size=2, *, palette=None)

   Draw the saved spot UMAP table, optionally colored by domain labels. With
   no supplied axes, use a 75 by 48.35 mm canvas, point area 2 points squared,
   opacity 1, and a two-column domain legend inside the canvas.

.. py:function:: seareg.plotting.joint_umap(table, labels=None, title="SEAREG (spots + peaks + motifs)", ax=None, point_size=5.2, *, peak_size=2.4, motif_size=4.2, annotations=None, text_expand=(1.05, 1.2), text_max_leader_mm=6, palette=None)

   Draw every saved joint UMAP node on a 75 by 113.2 mm canvas when no axes are
   supplied. Peaks use ``#CFCFCF`` triangles with area 2.4 points squared and
   opacity 0.22; motifs or kmers use ``#2C2C2C`` squares with area 4.2 points
   squared and opacity 0.20. Spot circles use area 5.2 points squared and
   opacity 0.65. The legend identifies node types. ``labels`` supplies the spatial
   domain labels for the spot subset only. A DataFrame with ``spot_id`` and
   ``domain`` is matched by ID and must cover every spot exactly once;
   missing, extra, or duplicated IDs are rejected. A one-dimensional array
   must follow the order of the joint table's spot rows. Without labels,
   spots use one blue color.

   Plotting uses the saved two-dimensional coordinates without manual
   rotation, reflection, or alignment. ``point_size`` controls the spot area;
   ``peak_size`` and ``motif_size`` control their respective node areas.
   These values are Matplotlib scatter areas in points squared, not radii.
   Optional ``annotations`` require ``entity_type``, ``entity_id``, and
   ``display_domain`` columns. Centered text starts at the matching typed node's
   current coordinates. The label-placement helper chooses nearby text boxes
   using rendered dimensions and a constrained SciPy optimization. It avoids
   text overlaps, the legend and leaders crossing other labels. ``text_expand``
   pads collision boxes. ``text_max_leader_mm`` bounds the visible leader length
   in millimeters (default 6); a larger bound allows more room for dense labels
   without moving the nodes. Infeasible layouts raise an error. Labels use their
   display-domain colors and leaders connect them to the current typed nodes.
   Tutorial 1 supplies 14 named reference features and retains those present
   in the current joint table. Feature labels share the ``feature`` category
   color; spot colors follow the current domain partition.

.. py:function:: seareg.plotting.save_figure(fig, stem, dpi=450, formats=("png", "pdf", "svg"), *, bbox_inches="tight")

   Save the figure in the requested formats, using ``stem`` as the filename
   prefix (string/``Path``); return a tuple of output ``Path`` objects.
   PDF/SVG preserve vector artwork and editable text. Existing output targets
   raise an error. The figure remains open for interactive editing. Set ``bbox_inches=None`` to preserve the figure's
   physical canvas dimensions; the spot and joint UMAP exports in T1–T5 use
   this option with ``dpi=300``, while T6 uses ``dpi=450``. Spatial exports
   retain the 450 DPI default.
   The default ``"tight"`` crops to the figure's tight bounding box.

Named plotting tables are aligned by IDs and checked for matching cohorts.
Joint nodes use the pair ``(entity_type, entity_id)``. Plain NumPy coordinate
arrays use positional order, so keep their rows aligned with the supplied
labels. Plotting functions do not change the global Matplotlib backend.
