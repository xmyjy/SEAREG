Continuous artifact pipeline
============================

The workflow API uses one artifact-driven execution path. The
original count matrix is decoded and peak-filtered once. Subsequent stages
consume immutable filtered counts, sequences, sites, selected edges, compiled
graphs and fixed training inputs. CLI, Python and compatibility entry points
use the same implementations.

This page describes the generic workflow API and its keyed artifact layout.
The five single-section notebooks configure this same ``Workflow`` from their first-cell
options, inspect its preparation stages, and launch a detached training job.
Their run and analysis layout is shown in :doc:`workflow`.
T6 uses the dedicated joint-section trainer described in :doc:`multislice`.

Guided counts-to-training entry
-------------------------------

Use the one-call interface to configure hyperparameters and automatically
complete all five preparation/training stages:

.. code-block:: python

   import seareg as sr
   training = sr.run_workflow(
       'data/new_dataset/counts.h5ad', output='result/new_dataset', motif=False,
       hidden_dim=64, lr=0.001, seed=1234,
       spatial_weight=0.5, dna_weight=0.1, pn_r=10,
       device='cpu',
   )
   print(training.path)

``run_workflow`` is a convenience adapter to
``Workflow.from_counts(...).run()``. ``training`` accepts a TrainConfig
(including a ``training_profile`` result) or dictionary for every other option;
explicit flat arguments override its matching fields. ``network`` accepts
NetworkConfig or a dictionary, for example ``{'use_spatial': False}``.
``physical_gpu`` overrides the matching execution mapping field. The caller's
configuration is copied before applying overrides.

For new schema-v2 workflows, omitted ``epochs`` and ``shards`` are resolved
from the prepared network: 500 epochs when the total training spot count is
greater than 2,500, otherwise 300; ``max(10, ceil(E / 700000))`` shards,
where ``E`` is the filtered binary count matrix's number of distinct positive
spot–peak pairs. Shards are integers, with no rounding to multiples of ten.
Explicit arguments or ``training`` fields override the corresponding automatic
value. Passing a complete ``TrainConfig`` or ``training_profile`` result makes
all its fields explicit. Existing tutorial configurations retain their settings.

By default the result is TrainingRun. Supplying an AnalysisConfig explicitly
also requests the existing analysis/render pipeline and returns AnalysisResult.
These one-call Python and CLI entries run synchronously: keep the calling
process alive until they return. For long training runs, use the independent
tmux job interface shown below.
All artifact paths are indexed by ``<output>/workflow.json``. Repeating a call
with matching input/configuration/code reopens completed work; a changed
hyperparameter branches the affected downstream stages under the same output
root. An explicit ``resume_from`` must satisfy the existing checkpoint guards.

For CUDA pass both ``device='cuda:0'`` and ``physical_gpu=<chosen card>``.
The :doc:`cli` page documents the equivalent direct-counts command, all
accepted options and its execution prerequisites.

Use ``motif=True`` and the actual assembly for automatic R annotation as
described below. See ``examples/run_full_workflow.py`` for a runnable example.

``Workflow.from_counts`` needs no JSON file. ``motif`` is a strict Python
boolean; JSON uses ``"motif": true/false`` and CLI uses ``--motif true/false``.
The default is true, matching the original model's motif-enabled default.
Older JSON configurations without a ``motif`` key retain their existing
counts-only or supplied-DNA/annotation behavior; no new scan is inferred for
an old counts-only configuration. ``examples/new_dataset_auto.json`` sets the
switch explicitly.

.. code-block:: python

   import seareg as sr
   flow = sr.Workflow.from_counts('data/new_dataset/counts.h5ad', output='result/new_dataset', motif=False)
   filtered = flow.filter()
   selected = flow.prepare_motif()       # None for motif=False; no R/reference access
   network = flow.prepare_network()
   inputs = flow.prepare_training()
   training = flow.train()

Each step can be inspected through its returned handle before continuing.
It saves a separate immutable artifact and reopens matching completed work.
``train`` is the first step that optimizes model parameters. ``prepare_network``
compiles static sparse graphs, and ``prepare_training`` assembles/fixes the
train/test split rather than optimizing a model.

For training that continues after the notebook or SSH session closes, install
tmux and replace the synchronous ``flow.train()`` call with:

.. code-block:: python

   job = sr.start_training_job(
       inputs, flow.config['training'], 'result/new_dataset/detached_training',
       execution=flow.config['execution'], name='new_dataset',
   )
   print(job.path / 'TRAINING_HANDOFF.md')
   training = job.wait()

The training output and its sibling ``detached_training_job/`` must be unused.
Reconnect with ``sr.load_training_job(job.path).wait()``; the job's handoff
records logs, checkpoints and its recovery script. This direct submission is
separate from ``Workflow``'s keyed training stage: pass the returned
``TrainingRun`` explicitly to ``flow.analyze(training=training)`` if analysis
is configured. See :doc:`workflow` for recovery.

For motifs, choose ``motif=True`` and optionally ``motif_source``:

* ``auto``: read an existing named matrix, or automatically invoke the original
  R scanner if absent;
* ``read``: require an existing matrix, never scan;
* ``scan``: use the original R scanner even if a nearby matrix already exists.

Automatic discovery checks ``motifs.h5``, ``freq_motif.h5`` and
``output_kmers_motifs/freq_motif.h5`` beside the counts file; ``dna=`` supplies
an explicit path. New R outputs are kept in the workflow artifact tree, not
written over the caller's input files. They use the graph recipe in
:doc:`motif_annotation`.

Scanning requires ``genome=`` or scalar ``uns['genome']``/``uns['assembly']``
metadata. A local matching ``fasta=`` can be supplied. For declared ``mm10``,
``mm39``, ``hg19`` and ``hg38``, the entry can prepare a missing reference from
the official UCSC download/checksum directory. Downloads occur only at the
annotation step, not ``filter``, ``check`` or ``plan``. The downloaded reference
has a local SHA-256 receipt and is validated on reuse. ``download_reference=False``
disables downloads. No assembly is guessed from chromosome names.

``genome`` selects the reference assembly; motif species are a separate choice.
Automatic R scanning uses the bundled JASPAR recipe's default selection of
``Homo sapiens`` and ``Mus musculus``. For a new scan with a different selection,
set ``motif.species_names`` in the workflow JSON or use the explicit annotation
API in :doc:`motif_annotation`. That page also describes the bundled
scanner's vertebrate fallback. When a new scan is needed,
``Workflow.from_counts`` uses the default species selection regardless of
``genome``. Changing the motif resource
selection changes the training input and should be recorded with the run.

R/bedtools/Bioconductor are optional installation dependencies, provided by
``environment-motif.yml``. They are not installed silently at training time.
The original R/JASPAR2024 backend needs its database accessible to the R
package. Missing or broken resource access is a failed annotation, never an
automatic fallback to motif=False or a different motif database.
The automatic entry uses ``<output>/resources/BiocFileCache`` for the R
database, seeding only an existing readable JASPAR2024 cache entry when possible.
Otherwise the original R package prepares its database there. An explicit
``BFC_CACHE`` environment variable is respected. Home caches and user FASTA
indexes are not modified. ``motif=False`` does not prepare these resources.

Stage APIs
----------

.. code-block:: python

   import seareg as sr

   filtered = sr.filter_counts('data/new_dataset/counts.h5ad', sr.FilterConfig(), 'result/new_dataset/filtered')
   sequences = sr.prepare_sequences(
       filtered, sr.SequenceConfig('/reference/genome.fa', 'explicit-build'),
       'result/new_dataset/sequences',
   )
   annotation = sr.annotate_motifs(
       sequences,
       sr.AnnotationConfig('/motifs/resource.meme', fimo='/tools/fimo'),
       'result/new_dataset/annotation',
   )
   selected = sr.select_motif_edges(
       annotation, sr.EdgeSelectionConfig('all_candidates'), 'result/new_dataset/selected',
   )
   network = sr.build_network(filtered, selected, sr.NetworkConfig(), 'result/new_dataset/network')
   config = sr.training_config(network, device='cpu')
   inputs = sr.prepare_training(network, config, 'result/new_dataset/training_inputs')
   job = sr.start_training_job(inputs, config, 'result/new_dataset/training',
                               execution={'physical_gpu': None}, name='new_dataset')
   training = job.wait()

``all_candidates`` selects every candidate relation. ``site_p`` requires a
cutoff no larger than the scan threshold. Empty annotations and selections can
be saved; network preparation requires a nonempty selected motif graph. See
:doc:`unified_framework` for method interpretation.

``training_config`` reads validated filtered spot and positive-edge counts
from the prepared network and returns a ``TrainConfig``. For existing filtered
statistics, ``sr.training_defaults(n_spots, positive_edges=E)`` returns the same
epoch/shard choices. The edge count uses each retained nonzero pair once,
regardless of count magnitude; motif and spatial edges are excluded.

For joint sections, resolve the configuration before creating immutable stages:

.. code-block:: python

   counts = 'data/new_sections/counts.h5ad'
   config = sr.multislice_training_config(
       counts, device='cpu', correction_mode='adaptive_depth_gate')
   filtered = sr.filter_multislice_counts(counts, config, 'result/new_sections/filtered')
   network = sr.build_multislice_network(filtered, config, 'result/new_sections/network')
   inputs = sr.prepare_multislice_training(network, config, 'result/new_sections/training_inputs')
   job = sr.start_multislice_training_job(
       inputs, config, 'result/new_sections/training', execution={'physical_gpu': None})
   training = job.wait()

The helper reads the actual section IDs and applies the existing multislice
peak filter when counting edges. Each section gets its own
``max(10, ceil(E_section / 700000))`` shards. Epochs use the joint total spot
count, and ``updates_per_epoch`` is the largest final section shard count.
Explicit helper overrides take precedence. See :doc:`multislice` for the joint
model, coordinate-coverage preflight and adaptive_depth_gate profile.

Each creation call requires an unused output directory. Load completed work
with ``FilteredCounts.load``, ``PeakSequences.load``, ``AnnotationResult.load``,
``SelectedMotifEdges.load``, ``ModelNetwork.load``, ``TrainingInputs.load`` or
``load_run``. Loading a handle verifies identities without decoding its complete
count matrix. Numeric data is materialized explicitly. Complete workspaces can
be moved together: dependencies are relative, content-bound references.

What is saved
-------------

``FilteredCounts``
  ``counts.h5ad`` retains selected original counts, metadata and coordinates.
  ``peak_filter.tsv.gz`` records every original peak, frequency, decision,
  reason and final index. ``filter_report.json`` records the exact rule and
  cohort counts. Training uses a binary CSR view of this fixed nonzero structure.

``PeakSequences``
  ``peaks.fa``, ``peaks.bed``, ``peak_sequences.tsv`` and ``background.json``
  freeze the peak order, intervals, sequences and pooled complement-symmetric
  A/T and C/G background. Reference and optional FASTA-index hashes bind the
  explicitly declared assembly. No genome or motif database is downloaded.

``MotifAnnotation`` / ``AnnotationResult``
  ``motif_sites.tsv.gz`` preserves all reported strand-specific sites, raw
  score/P text, matched sequence, relative and genomic coordinates.
  ``peak_motif_relations.tsv.gz`` preserves complete candidate pairs with best
  site P, physical-site and strand-site counts. ``motif_ids.tsv`` and
  ``peak_ids.tsv`` retain the complete axes, including zero-hit status.
  ``load_matches()`` derives an in-memory sparse view of the saved tables.

``SelectedMotifEdges``
  ``selection.json`` and ``selection_decisions.tsv.gz`` record the rule,
  thresholds and decisions. Non-identity selections save
  ``selected_edges.tsv.gz``. Identity selection references the complete
  annotation relation table instead of copying it. ``edges()`` always yields
  the named selected-edge view. Imported matrices are marked as lacking
  original site evidence.

``ModelNetwork``
  ``effective_motif_edges.npz`` contains the model's actual sparse matrix after
  the configured binary/weighted, empty-column and duplicate-neighborhood rules.
  ``motif_node_map.tsv.gz`` and ``motif_edge_lineage.tsv.gz`` explain the mapping
  from original motifs/selected relations to model nodes and edges. Duplicate
  columns select the existing representative; their weights are not summed.
  ``spatial_adjacency.npz``, ``motif_adjacency.npz`` and ``degrees.npz`` preserve
  the actual coalesced graph values and raw degrees. Training loads them without
  recomputing neighborhoods, aligning motifs or renormalizing static graphs.

``TrainingInputs``
  ``split.npz`` preserves ordered train positives, test triplets and the initial
  shard permutations. ``full_train_adjacency.npz`` contains only training
  positives. The tutorial profiles use ``tr=1``; see
  :doc:`unified_framework` for evaluation scope.

``TrainingRun``
  Stores training history, exported embeddings and ordinary checkpoints.
  Checkpoints retain model and Adam state, compiled input identity, and complete
  random/sampler state. Resume uses the seed/epoch sampling streams.

Read saved results, without rescanning
--------------------------------------

.. code-block:: python

   import seareg as sr

   annotation = sr.AnnotationResult.load('/workspace/artifacts/annotation/STAGE_KEY')
   pairs = annotation.relations()
   selected = sr.SelectedMotifEdges.load('/workspace/artifacts/selected_edges/STAGE_KEY')
   named_edges = selected.edges()
   network = sr.ModelNetwork.load('/workspace/artifacts/network/STAGE_KEY')
   model_nodes = network.motif_node_map()
   model_edges = network.edge_lineage()

The immutable upstream artifacts must be available, but the original FASTA,
raw count source and scanner executable are not needed to read an already
completed annotation or network.

To select another graph, load the annotation and call ``select_motif_edges``
with a new output directory. To train again, load ``TrainingInputs`` and call
``train_SEAREG`` with a new run directory. An explicit compatible checkpoint can
be supplied as ``resume_from``. See :doc:`architecture` for checkpoint
compatibility.

Unified workflow and command line
---------------------------------

``examples/new_dataset_fimo.json`` shows schema v2. Set the raw count path,
reference/assembly, motif resource and scanner, then declare an edge-selection
rule before training. Schema-v1 configurations retain their numerical settings
when loaded.

Use :doc:`cli` for configuration-mode commands, direct-input flags and stage
restrictions. Start with ``check`` and ``plan``; both are non-executing and
create no output directory. Uncomputed upstream stages are pending.
``analyze`` requires completed training; ``run`` executes required upstreams.

Created artifacts are under ``artifacts/STAGE/STAGE_KEY``; training runs are
under ``runs/RUN_KEY``, numerical analyses under ``analyses/ANALYSIS_KEY`` and
fit-free figures under ``figures/RENDER_KEY``. ``workflow.json`` records the
latest stage references, while per-stage receipts and resolved configurations
preserve each branch. A different edge rule or training configuration gets a
different branch instead of overwriting completed results.

Numerical analysis and rendering are separate. ``Workflow.run()`` returns the
numerical ``AnalysisResult`` when analysis is configured. Its saved graph,
domains and UMAP can be rendered through ``Workflow.render(result)`` without
fitting again. Root ``plots=false`` skips automatic rendering. Changing only
plotting implementation/style does not invalidate filtering, FIMO or training.

GPU and resource settings
-------------------------

For GPU training, declare worker ``training.device="cuda:0"`` and the chosen
physical card in ``execution.physical_gpu``. The new-dataset template defaults
to CPU. See :doc:`unified_framework` for loss-workspace, caching and annotation
resource settings.

Compatibility
-------------

The legacy R scanner extracts its own sequences and produces its selected
training graph. Its adapter bypasses the FIMO sequence-artifact stage.
Earlier ``scan_peak_motifs_legacy``, ``annotate_peak_motifs``,
``import_fimo_annotation``, ``export_motif_graph`` and manual preparation APIs
route to the same stage implementations. HDF5 training matrices are written
only for an explicit compatibility export.
