Generate peak–motif associations
================================

This page documents the JASPAR2024/motifmatchr training-graph scanner. For
FIMO site annotation, explicit graph selection and method interpretation, see
:doc:`unified_framework`.

SEAREG includes a JASPAR2024/motifmatchr scanner, packaged as
``seareg/scanners/scan_for_kmers_motifs_new_2024.R`` and is available through
``seareg.scan_peak_motifs_legacy`` and the ``seareg-annotate`` command. An
existing ``motifs.h5`` is not required to run this stage.

The five single-section tutorials accept the selected ``MOTIF_PATH`` through the same
interface for examples and user data. A bundled archive can be unpacked during
input preparation. After peak filtering, ``flow.prepare_motif()`` imports or
scans annotations according to ``MOTIF_OPTIONS`` and validates their contents
and coverage. An invalid selected matrix raises an error. Filtering first
fixes the sequence-extraction axis, scanner subject background and
proportional Top-K calculation.

In the first notebook cell, set ``MOTIF_OPTIONS["source"]`` to ``"read"``,
``"auto"`` or ``"scan"``. The ``"auto"`` setting reads an existing matrix
or scans when one is unavailable. ``enabled=False`` disables motif preparation.
For scanning, choose the actual ``genome`` and ``species_names`` and optionally
a matching local ``fasta``. The visible species default is ``Homo sapiens``
and ``Mus musculus``. ``rscript`` selects the executable, and
``download_reference`` controls supported reference downloads; see
:doc:`continuous_pipeline`.

``SEAREG_REFERENCE_ASSEMBLY``, ``SEAREG_REFERENCE_FASTA``,
``SEAREG_MOTIF_SPECIES`` (comma-separated names) and ``SEAREG_RSCRIPT`` can
supply initial configuration values. Existing annotations need no R or
reference resources. Generated annotations record the selected reference,
species and scanner provenance in the new run directory.

Dependencies
------------

Install the :doc:`Python runtime <installation>`, bedtools, R, and these R
packages before scanning:

* ``optparse``, ``Matrix``, ``DBI``, ``RSQLite``;
* ``rhdf5``, ``HDF5Array``, ``Biostrings``, ``TFBSTools``, ``JASPAR2024``,
  ``motifmatchr``, ``SummarizedExperiment``;
* ``BiocParallel`` when requesting more than one scanner worker.

Initialize the JASPAR2024 database cache from the
`official 2024 archive <https://jaspar2024.elixir.no/downloads/>`_ once after
installing the R environment. This uses the cache key expected by the installed
R package and avoids relying on its older download mirror:

.. code-block:: bash

   "$SEAREG_RSCRIPT" --vanilla - <<'RS'
   library(BiocFileCache)
   library(JASPAR2024)
   metadata <- read.csv(system.file("extdata", "metadata.csv", package="JASPAR2024"))
   requested <- metadata$SourceUrl
   cache <- BiocFileCache(ask=FALSE)
   matches <- bfcquery(cache, requested, field="rname", exact=TRUE)
   stopifnot(nrow(matches) <= 1L)
   if (nrow(matches) == 0L) {
       bfcadd(cache, rname=requested,
              fpath="https://jaspar2024.elixir.no/download/database/JASPAR2024.sqlite",
              rtype="web")
   }
   database <- bfcrpath(cache, requested)
   stopifnot(length(database) == 1L, file.exists(database))
   JASPAR2024::JASPAR2024()
   RS

Repeated setup reuses the cached database. ``XDG_CACHE_HOME`` can select a
writable cache location. The workflow copies readable JASPAR entries into its
own resource cache; it does not modify the source cache.

Supply a local,
uncompressed reference-genome FASTA, using the same genome
assembly and BED coordinate convention as the input peaks. The explicit
``scan_peak_motifs_legacy`` API requires an assembly tag and records the FASTA
SHA-256. That adapter does not infer an assembly or download a replacement
reference. ``Workflow`` can prepare a supported reference when its configured
download option permits it.

Python workflow
---------------

Filter once, then pass the resulting ``FilteredCounts`` directly to the R
adapter. Network construction reuses the same filtered artifact:

.. code-block:: python

   from pathlib import Path
   import seareg as sr

   counts = Path("data/my_section/counts.h5ad")
   filtered = sr.filter_counts(counts, sr.FilterConfig(), Path("result/my_section/filtered"))
   annotation = sr.scan_peak_motifs_legacy(
       filtered,
       fasta=Path("references/mm10.fa"),
       assembly="mm10",
       output=Path("result/my_section/motif_annotation_v1"),
       species_names=("Homo sapiens", "Mus musculus"),
       p_cutoff=1e-5,
       threads=1,
       rscript="Rscript",
   )
   selected = sr.import_motif_edges(annotation / "freq_motif.h5", Path("result/my_section/motif_edges"))
   network = sr.build_network(
       filtered, selected, sr.NetworkConfig(), Path("result/my_section/network")
   )

``filter_counts`` reads the H5AD and applies ``copy1_frequency_v1`` once.
The annotation call writes a three-column BED in that exact final peak order
and runs the bundled R source; it does not filter the original counts again.
Choose the reference and species explicitly for the dataset. For example,
``species_names=("Homo sapiens",)`` requests human matrices; the bundled
scanner adds its vertebrate fallback when fewer than 300 matrices are found.

If the final peak axis is already saved as a ``PreparedNetwork`` or
``PreparedData``, pass that object or the prepared-network directory instead
of ``counts``. Omit ``preprocessing`` in that case, because its peak selection
has already been fixed. This allows scanning a network prepared without DNA
associations. Use a new input/network output directory to attach the generated
matrix; completed artifacts are not edited in place.

Command line
------------

The annotation command does not require a training configuration:

.. code-block:: bash

   seareg-annotate \
     --atac data/my_section/counts.h5ad \
     --fasta references/mm10.fa \
     --assembly mm10 \
     --output result/my_section/motif_annotation_v1 \
     --species "Homo sapiens" "Mus musculus" \
     --p-cutoff 1e-5 --threads 1

Use ``--prepared path/to/prepared`` in place of ``--atac`` for a fixed peak
axis. ``--rscript /path/to/Rscript`` selects an R environment. For a counts
matrix without spatial coordinates, pass ``--no-spatial``; use the same
setting when preparing the training network.

What the R scanner produces
---------------------------

The scanner produces a pruned training graph with the following recipe:

1. Query JASPAR2024 CORE for the requested species and combine unique motif
   IDs, retaining the original fallback behavior.
2. Run motifmatchr with the specified ``p_cutoff`` and its subject-sequence
   background. The compatibility entry point uses ``p_cutoff=1e-5``; calling
   the original R file directly uses its own default of ``5e-5`` unless
   overridden.
3. Compute ``log1p(nonnegative score) * (1 + log1p(count))``.
4. Keep motifs occurring in at least 20 peaks and at most 30% of peaks.
5. Retain the top 4% of peaks per motif, with K bounded to 500–5000, then
   retain at most 30 motifs per peak. Preserve the original input-order
   handling of tied values. Do not apply IDF.

The scanner's weighted matrix is saved before the training loader's binary
conversion, removal of empty columns, or optional motif-neighborhood
deduplication. Scanner motif columns and training motif nodes can therefore
have different counts.

The packaged ``seareg/scanners/legacy_motif_recipe.json`` records the scanner
settings.

Outputs and validation
----------------------

The returned directory contains ``freq_motif.h5`` with named
``mat/row_names/col_names``, ``peaks.bed``, the extracted sequences,
``sessionInfo.txt``, scanner logs, ``scanner.provenance.json`` and
``manifest.json``. Provenance binds the input, final peak order, declared
assembly, FASTA hash, scanner source hash and effective arguments. The adapter
validates matrix dimensions, nonnegative finite values, and exact peak order
before publishing the new directory. A failed scan does not publish a
completed output, and an existing output is never overwritten.

For another scanner that implements the same BED/FASTA/HDF5 contract, use
``seareg.scan_annotation`` with an explicit executable argument list. This
generic adapter is separate from the bundled R recipe.
