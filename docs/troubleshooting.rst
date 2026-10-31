Troubleshooting
===============

Use the issue that matches the stage you are running. These checks do not
require changing the scientific profile.

The workflow fails on Windows or macOS
--------------------------------------

Run the scientific workflow on Linux x86_64; Windows users can use Linux
under WSL2. Native output publication uses Linux ``renameat2`` and POSIX
locks. Browser reading and docs-only HTML builds are separate from that
runtime requirement. Follow :doc:`installation` for the pinned environment
or :doc:`hosting` for a documentation build.

The first cell cannot find ROOT or an input file
------------------------------------------------

Launch Jupyter from the complete checkout, where ``seareg/`` and
``tutorials/`` are present. Wheels and source distributions omit example
data and notebooks. Check ``ATAC_PATH`` and ``EMBEDDING_DIR`` as described in
:doc:`quickstart`. Prepare a bundled input explicitly with
``python tools/prepare_data.py --dataset ME13_joint_2187`` for T1.
Only exact included paths trigger automatic archive extraction; a missing
custom path does not select a replacement dataset. Use the manifest and
dataset-specific paths in :doc:`data_access` for another tutorial.

Counts, motifs or provided embeddings do not match
--------------------------------------------------

Read the named file/ID in the validation error. Check unique spot IDs,
genomic intervals, counts and ``obsm["spatial"]`` against :doc:`input_data`.
For motifs, use the same genome assembly and coordinate convention as the
counts; the tutorial's strict alignment rejects missing annotations for
retained peaks. ``filter_report.json`` identifies rows that become empty
after filtering. Resolve those inputs explicitly before training.

For provided mode, select the directory containing ``export.json`` and all
its referenced TSVs, together with the matching counts cohort. Preserve the
receipt and files as a group; do not edit IDs or hashes to bypass validation.
An export marked ineligible for analysis needs an eligible propagated export.
See :doc:`workflow` for intermediate-export behavior.

My domains or UMAP differ between environments
----------------------------------------------

First confirm the same inputs, export source, profile and numerical runtime.
Compare ``tutorial_execution.json``, the analysis manifest and
``analysis/spot_graph_metadata.json``. The latter records the actual neighbor
backend under ``neighbor_execution``. A pinned environment and seeds do not
guarantee identical results across hardware or numerical implementations.
``sc.pp.neighbors`` constructs the graph; ``sc.tl.louvain`` consumes that
graph, so compare graph provenance before interpreting a partition change.

To inspect the exact existing figure, reopen its saved analysis and use its
coordinates/labels rather than recomputing UMAP or domains. See
:doc:`quickstart` and :doc:`workflow`. Do not change K, resolution or neighbor
settings merely to reproduce a display.

I disconnected while a training job was running
------------------------------------------------

Read that job's ``TRAINING_HANDOFF.md`` first, then check the recorded tmux
session, live process, GPU when used, logs, diagnostics and checkpoint times.
The notebook waiting for a job does not own its worker. To reconnect to the
existing job from Python, use its actual job directory:

.. code-block:: python

   import seareg as sr

   training = sr.load_training_job("/absolute/path/to/training_job").wait()

If the worker stopped, inspect its exception and last complete epoch before
using the generated ``resume.sh``. Recovery checks input, implementation and
numerical identities. See :doc:`workflow` for recovery requirements; a
connection loss alone is not a reason to launch another training run.

Can the included embeddings resume training?
--------------------------------------------

Provided exports support downstream analysis. Exact continuation needs the
compatible full-state checkpoint, including optimizer, RNG and sampler
state. A TSV export or model weights alone cannot supply it. Complete
checkpoints are not distributed with the examples. The asset table in
:doc:`quickstart` distinguishes export, run and analysis loaders.

How do I reopen results or save an editable figure?
---------------------------------------------------

Load the saved analysis with ``seareg.analysis.load_result`` and its
``domains()``, ``spot_embedding()`` or ``joint_embedding()`` methods, as
appropriate for the outputs enabled in that analysis. These readers do not
fit again. ``Workflow.render(result)`` renders an existing numerical result;
see :doc:`continuous_pipeline`.

The tutorial export cells use ``seareg.plotting.save_figure`` to save PNG
and vector PDF/SVG in the run's ``figures/`` directory. Reuse saved numerical
coordinates and labels for display changes, disable grid lines through the
data area, and save through this plotting path. Increasing PNG DPI or placing
a notebook screenshot in a PDF does not produce editable vector elements.
See :doc:`api` for the figure-saving interface.

Multislice four-side coverage or alpha is zero
----------------------------------------------

Inspect native coordinates and the original within-section graph with
``seareg.multislice.geometry.spatial_coverage_report``. Eight-sector boundary
coverage and unit four-orthogonal-side coverage are reported separately per
section. Micrometer units, rotated grids, hexagonal layouts or irregular
points can have no valid four-side evidence. Do not rescale, rotate, substitute
kNN or reuse display transforms automatically: each changes the model input.
Invalid rows get zero additional correction and retain the established coherence correction.

Geometric preflight cannot compute alpha. After a full-message refresh, read
the runtime coverage, denominator, ``Q=O*D``, alpha and budget diagnostics.
Zero remaining capacity, invalid donor mass, zero displacement or a zero
section denominator can yield alpha=0. A valid Q=0 row can still receive
``e=alpha_s*H*V`` from its section's allocation; the model does not apply a
pointwise 3% Q coefficient. See :doc:`multislice` for the matched-budget rule.

A multislice checkpoint is rejected
-----------------------------------

Use a compatible native checkpoint with matching input/coordinate,
section, formula, implementation and numerical identities. Do not bypass
identity checks or call weight-only
loading exact recovery. Read the job handoff and use its generated resume
script; complete state includes all gate/geometry/protection/coherence/displacement-budget
caches, Adam, RNG and sampling order. Saved analysis and node exports support
result inspection, not full-state recovery.
