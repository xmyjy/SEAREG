Input format and preprocessing
==============================

Each single-section application provides a count matrix, spatial coordinates
and optional peak–motif associations. The configuration key ``input.atac`` is used
for both ATAC and CUT&Tag H3K27ac spot-by-peak matrices. For the co-profiled
datasets, only the indicated chromatin modality enters training; the RNA
modality is not required.

For joint sections, T6 requires section IDs in ``obs["sample_id"]`` and shared
genomic peak identities. Its separate tissue annotation TSV is used only for
display. See :doc:`multislice` for this input schema and its per-section
preparation thresholds.

Count matrix
------------

Use an AnnData H5AD file with the following fields:

.. list-table:: ``counts.h5ad``
   :header-rows: 1
   :widths: 25 75

   * - Field
     - Content
   * - ``X``
     - Spots by peaks; finite, nonnegative counts. Dense and sparse matrices are accepted.
   * - ``obs_names``
     - Unique, nonempty spot IDs, stored as text.
   * - ``var_names``
     - Unique, nonempty peak IDs. Retained chromosome features must identify genomic intervals.
   * - ``obsm['spatial']``
     - Numeric spatial coordinates in the same row order as ``X``, with at least two columns and finite values.

Coordinate units should be consistent within the section. The plotting
interface keeps the original coordinate values and increasing y direction.
Text IDs are preserved, including leading zeros.

For multislice cs_matched03, coordinate units and geometry also determine
microgain coverage: valid evidence requires all four orthogonal sides along
unit, axis-aligned edges already in the native within-section spatial graph
(tolerance 1e-6). Micrometer units, rotated grids, hexagonal layouts and
irregular points may yield zero coverage. Inspect the per-section
``spatial_coverage_report`` (see :doc:`api`) for both eight-sector boundary
coverage and four-side microgain coverage. Invalid microgain rows get zero
extra term and retain the CS path. Alpha and budgets are available only after
full-message refresh, not geometric preflight. No silent rescaling, rotation
or kNN substitution is applied; display transforms never become training
coordinates. Shared genomic peak IDs must use a common axis across sections,
with each shared peak represented once in the joint model.

Motif associations
------------------

For a dataset without an existing association matrix, use the bundled
:doc:`peak–motif generator <motif_annotation>` before network preparation.
``seareg.scan_peak_motifs_legacy`` preserves the original JASPAR2024/motifmatchr
training-graph recipe and produces ``freq_motif.h5`` in the format below.

The standard ``motifs.h5`` file is an HDF5 file with three datasets:

.. list-table:: ``motifs.h5``
   :header-rows: 1
   :widths: 25 75

   * - Dataset
     - Content
   * - ``mat``
     - A numeric matrix with peaks in rows and motifs in columns.
   * - ``row_names``
     - Unique, explicit genomic interval IDs for the matrix rows.
   * - ``col_names``
     - Unique, explicit motif IDs for the matrix columns.

An H5AD annotation file is also accepted: ``X`` contains peaks by motifs,
``obs_names`` names the peaks and ``var_names`` names the motifs. The annotation
matrix must have finite, nonnegative values. The tutorial configurations use
``dna_kind='motif'``, ``dna_weight='binary'``,
``motif_deduplicate=True`` and ``allow_partial_dna=False``.

Peak matching
-------------

Both matrices are matched by genomic interval, after parsing their peak names.
These names describe the same matching key:

.. code-block:: text

   chr1:100-200
   chr1_100_200
   chr1-100-200
   chr1_100-200

The parser requires a ``chr`` chromosome prefix, nonnegative integer
coordinates and an end greater than the start. The motif rows are reordered
to match the retained count-matrix peaks. A duplicate canonical interval or
missing motif annotation for a retained peak raises an error under the tutorial
settings. Original count-matrix peak IDs are retained in the prepared data.

Both inputs must use the same genome assembly and coordinate convention.
Parsing a different separator does not convert a genome assembly or a
0-based/1-based coordinate system.

Filtering and binary graph inputs
---------------------------------

The current ``copy1_frequency_v1`` protocol counts how many spots have a
positive observation for each peak. For ``n`` input spots, it uses:

.. code-block:: python

   ratio = 0.01 if n <= 2500 else 0.005
   min_spots = max(1, int(n * ratio))
   max_spots = int(n * (1 - ratio))

A peak is retained if its name starts with ``chr`` and its frequency is between
these bounds, including both endpoints. Positive counts become one; zero
counts remain zero. The filter preserves spot order and does not remove spots
based on a dataset-specific list. The prepared output records the retained
IDs, dropped peaks and empty spot rows for inspection.

Every spot must retain at least one positive count after peak filtering.
An originally nonempty spot can become empty when all of its peaks are removed.
Filtering preserves the original spot axis and records ``empty_spot_ids`` in
``filter_report.json``. Training rejects these rows because the retained global
spatial-scale formula is undefined for their count support. Inspect the report
and explicitly resolve or exclude the affected spots before preprocessing.
After the fixed split, training checks the actual graph and effective final
spatial/motif branches for unsupported export nodes before optimization,
without changing the split.

New-data training defaults use this filtered binary graph: ``E`` counts each
distinct nonzero spot–peak pair once, independent of count magnitude. It does
not include removed peaks, motif edges or spatial edges. Single-section
workflows use ``max(10, ceil(E / 700000))`` shards. For joint sections, the
existing multislice filter is applied first and the same rule uses each
section's filtered edge count. The epoch default is 500 when the total training
spot count is greater than 2,500, otherwise 300; joint training uses the total
across sections. See :doc:`continuous_pipeline` for helpers and overrides.

Single-section input with exactly 2,500 spots uses the 1% frequency ratio
and a default of 300 training epochs.

For motif annotations, positive associations become one. Columns that are
empty after peak alignment are removed. With motif deduplication enabled,
columns with identical aligned associations share a representative; the
original motif-to-representative mapping is saved in the prepared data.

When checks happen
------------------

``Workflow.check()`` reads file structure, axis IDs, coordinate values and
content hashes, and checks dependencies. Full count-value validation and peak
filtering occur in ``Workflow.filter()``; ``Workflow.prepare_inputs()`` is an
alias for that stage. ``Workflow.prepare_network()`` reuses the filtered counts,
validates motif values and aligns the motif associations.

The separate compatibility function ``seareg.prepare_inputs(...)`` records
input file identities and returns an ``InputArtifact``. It does not filter
counts.

Schema v2 can reuse a prepared model network through ``input.network``;
schema v1 uses ``network.prepared``. Reused artifacts must match the supplied
inputs and configuration. The tutorials build a new network from their input
files.
