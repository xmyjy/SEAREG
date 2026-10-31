Method and training configuration
=================================

SEAREG connects tissue organization with chromatin regulatory features through
multi-relational heterogeneous graph representation learning. Spatial
neighborhoods, spot–peak observations and optional peak–motif associations
contribute to a shared representation of spots and genomic features.
The :doc:`continuous_pipeline` page describes stage APIs and JSON configuration.
This page explains the learned representations, manuscript applications,
training execution profiles and optional FIMO backend. The five single-section
tutorials read their supplied motif matrices;
the :doc:`R scanner <motif_annotation>` is their missing-annotation branch.
The motif-free T6 adaptive_depth_gate model and balanced joint-section protocol
are described in :doc:`multislice`.

Shared representations and feature profiles
-------------------------------------------

The sequence-informed single-section model jointly embeds spots, peaks and
TF motifs. Two LightGCN propagation layers combine the graph relations with
spatial residuals. The Bayesian personalized ranking objective ranks observed
spot–peak pairs above sampled unobserved pairs using dot products of their
propagated representations. Peak–motif links contribute through propagation.

For downstream feature scoring in the manuscript, nonzero embeddings are
L2-normalized and compared by cosine similarity. Spot–peak profiles supply
inferred spatial accessibility readouts; motif–spot profiles supply
motif-informed TF activity readouts. These peak profiles are association
scores rather than calibrated fragment counts. Motif–peak embedding similarity
(MPES) compares the two feature representations directly.

``sr.spatial_scores`` computes selected spot–peak and spot–motif cosine profiles
from the exported embeddings. Set ``standardize=True`` to obtain whole-cohort
z-scores (PSAS/MSAS), while retaining the raw cosine column. Tutorial 1 includes
one peak and one motif example; see :doc:`api` for the input and output contract.

In the E13 interface analysis, motif–spot and peak–spot profiles are
standardized across the section to obtain MSAS and PSAS. Their arithmetic mean
is the motif–peak composite spatial association score (MPcSAS), describing the
combined spatial pattern of the two features.

The P22 ATAC and H3K27ac sections are modeled independently before comparing
their regional chromatin programs. Candidate TF–element–target networks in
the manuscript combine embedding profiles, sequence matches, H3K27ac support,
downstream peak–gene links and TF/target RNA evidence. RNA contributes to these
downstream analyses rather than the chromatin-model inputs.

The developmental study uses motif-free joint spot–peak representations, with
shared peak identities and spatial neighborhoods confined to each section.
T6 implements joint analysis using adaptive depth gating: null-calibrated
donor weights, risk-modulated boundary protection, donor coherence and a
per-section reference displacement budget. It uses global negative sampling; see
:doc:`multislice` for the equations, coordinate limits and saved-result loading.
The notebooks cover training, domains
and UMAP; named spot/peak/motif
exports retain the identities needed for further feature-based analyses.

Data flow
---------

.. code-block:: text

   count matrix + spatial coordinates
                  |
   retained counts and peak IDs + optional named peak-motif associations
                  |
   spot-peak graph + spatial graph (+ peak-motif graph)
                  |
   two-layer LightGCN + spot-peak BPR + regularization
                  |
   checkpoints and named spot/peak/motif embeddings
                  |
   spatial domains / UMAP / downstream feature-profile analyses

Motif associations can be read from an existing matrix or prepared through
the R scanner or explicit FIMO annotation/selection stages. The latter are
described below.

The model uses two propagation layers. Spot and peak exports sum their first
and second layer representations; motif exports sum K1 and K2. Execution
backends can change floating-point reduction order and numerical trajectories.

Opt-in efficient training
-------------------------

.. code-block:: python

   import seareg as sr
   training = sr.training_profile(
       'balanced', epochs=300, device='cuda:0', seed=1234,
       loss_workspace_mb=1024, resident_cache_reserve_gib=4,
   )

Or add ``"training_profile": "balanced"`` at the JSON root. Explicit
``training`` entries override the preset, including scientific parameters.

* ``balanced``: grouped loss, weighted nodewise ego L2, resident caching and
  ``loss_backend="auto"`` with a 1024 MiB loss-workspace planning budget.
* ``speed``: the same helpers but explicit eager ``torch`` loss. This avoids
  checkpoint recomputation and may use substantially more memory.
* ``memory``: explicit ``torch_streamed`` checkpointed chunks and a 256 MiB
  loss-workspace planning budget.

``auto`` first uses CUDA fused sampled-dot scoring when supported by the active
runtime. Otherwise, it selects eager PyTorch when a whole shard fits the
conservative planning estimate and checkpointed chunks otherwise. Chunk
boundaries preserve complete ``pn_r`` groups. A budget too small for one group
is rejected. For the fused path, ``loss_workspace_mb`` is a PyTorch fallback
planning budget, not a limit on the fused kernel's workspace or total VRAM.
Explicit ``torch`` uses eager scoring and does not enforce this
budget; ``loss_chunk_size`` is available only for explicit ``torch``. There is
no dense spots-by-peaks score matrix in these loss paths.

The loss-workspace estimate is not a total VRAM guarantee: it excludes model,
optimizer, graph propagation, caching, allocator overhead and other allocations.
``resident_cache_budget_gib`` and ``resident_cache_reserve_gib`` independently
control the shard cache. Cache overflow uses the streaming fallback.

``TrainingRun.path / 'status.json'`` records requested/effective backend counts, actual
chunk-size histograms, cache payload and hits/misses, completed epochs, optimizer
steps, training wall time including setup/exports, and PyTorch allocator
allocated/reserved peaks on CUDA. Process-VRAM and CPU-memory peaks are null.
Per-epoch time excludes embedding/checkpoint exports; training wall time
includes them.

Complete local peak-motif annotation
------------------------------------

The :doc:`legacy scanner <motif_annotation>` is the default for
``seareg-annotate``. The FIMO branch requires a local executable, an
uncompressed local reference FASTA, a declared assembly and a local DNA MEME
resource. If a ``.fai`` exists it is used and hashed; otherwise a regular wrapped FASTA
is indexed in memory without changing the reference file.

.. code-block:: python

   from pathlib import Path
   import seareg as sr
   preprocessing = sr.PreprocessConfig()
   annotation = sr.annotate_peak_motifs(
       Path('data/new_dataset/counts.h5ad'), Path('data/new_dataset/reference.fa'), 'explicit-build',
       Path('data/new_dataset/motifs.meme'), Path('result/new_dataset/motif_annotation'),
       preprocessing=preprocessing, fimo='/absolute/path/to/fimo',
       p_cutoff=1e-5, motif_pseudocount=0,
   )
   # Choose a sequence-candidate training graph explicitly.
   graph = sr.export_motif_graph(
       annotation, Path('result/new_dataset/candidate_graph'), selection='all_candidates',
   )

The count input is filtered before sequences are extracted using zero-based,
half-open peak intervals. Final peak order and all MEME motif IDs are preserved,
including zero-hit rows/columns. The pooled A/T and C/G complement-symmetric
background comes from those exact sequences. FIMO scans both strands. Motif
matrix preparation is fixed by the supplied file and explicit pseudocount.
The artifact records FASTA, optional index, motifs, counts and scanner hashes,
version/arguments, site tables and pair aggregates. Canonical v2 annotations
store one site table and one relation table.
``load_matches()`` constructs an in-memory sparse view; H5 is an explicit
compatibility export only.

``motif_sites.tsv.gz`` preserves strand, score, site P-value, matched sequence,
relative coordinates and genomic coordinates. ``peak_motif_relations.tsv.gz``
records best site P, physical sites and strand-specific site counts per pair.
``peak_ids.tsv`` and ``motif_ids.tsv`` retain zero-hit status and names.
Every reported site is checked against the reference and its motif width.
``sr.import_fimo_annotation`` validates/imports a compatible existing FIMO text
table against the same axes and reference.

Site parsing is streamed and physical/strand deduplication uses a temporary
SQLite index. Axes and peak sequences remain in memory; plan for both memory
and temporary disk use.
The normalized site table retains the reported values; duplicate successful
raw stdout is removed after its hash is recorded. Failed attempts retain
their command, stderr and status outside the incomplete artifact.

Graph export requires ``selection='all_candidates'`` or ``selection='site_p'``
with an explicit ``p_cutoff`` no larger than the scanned cutoff. Both are
sequence-candidate graphs. An empty annotation remains valid; an empty
selected graph is rejected for training.
The graph retains full axes in sparse CSR HDF5; training preprocessing may
subsequently remove empty or duplicate motif neighborhoods according to the
unchanged configuration. Annotation motif count and model motif-node count
therefore need not match.

.. code-block:: bash

   seareg-annotate --method fimo --atac data/new_dataset/counts.h5ad \
     --fasta data/new_dataset/reference.fa --assembly explicit-build --motifs data/new_dataset/motifs.meme \
     --fimo /absolute/path/to/fimo --output result/new_dataset/motif_annotation

This command only annotates. Add both ``--graph-selection all_candidates`` and
``--graph-output result/new_dataset/candidate_graph`` to explicitly export a candidate graph.
For ``site_p``, also provide ``--graph-p-cutoff``. Neither command trains.

Unified workflow and examples
-----------------------------

``examples/new_dataset_fimo.json`` is a template. Replace every path/identity,
declare the domain target and scientific training parameters, then run
``check`` and ``annotate``. The template omits ``edge_selection``; add an
explicit graph rule to a finalized configuration before network preparation
or training.

.. code-block:: bash

   seareg --config examples/new_dataset_fimo.json check
   seareg --config examples/new_dataset_fimo.json annotate
   # After explicitly configuring edge_selection in a new finalized v2 config:
   seareg --config finalized.json run

``Workflow.check()`` inspects configuration/input structure without running a
scanner, training or analysis. ``Workflow.annotate()`` explicitly runs or
reopens the annotation stage. ``prepare-network``/``train``/``run`` include
configured annotation and explicit graph selection. Existing ``input.dna`` and
generated ``annotation`` are mutually exclusive. Scanner-specific options are
not interchangeable between the legacy and FIMO backends.

Completed workflow stages are reused only with matching input/config/code
identities and verified manifests. See :doc:`workflow` for the single-section tutorials
and their dataset-specific data and result directories.

Method interpretation
---------------------

Motif associations are sequence-based model inputs. They do not establish
transcription-factor occupancy, binding or regulatory causality. The legacy
scanner yields a pruned training graph; FIMO yields sites and candidate
relations at its scan threshold. ``all_candidates`` and ``site_p`` are graph
selection rules, not globally corrected confidence procedures. An imported
FIMO table is checked for consistency; its scan completeness remains the
caller's responsibility.

The tutorial profiles use ``tr=1.0``, so all positive spot–peak observations
enter training and no held-out positive evaluation is available. Domain
identification and UMAP describe the analyzed embeddings. Joint UMAP includes
spot, peak and motif nodes; proximity in this projection alone does not
establish a regulatory relationship.

The packaged ``spatial_scores`` function computes spot–peak and spot–motif
profiles. MPES/MPcSAS and candidate TF–element–target networks require
downstream analysis; they are not automatic pipeline stages.
