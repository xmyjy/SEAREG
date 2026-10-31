Developing mouse brain across sections
======================================

T6 uses the motif-free ``adaptive_depth_gate`` spot–peak model across four
MISAR-seq developmental sections. Adaptive gating reduces the influence of
spots with locally anomalous sequencing depth. The multislice profile is
``tutorials/configs/Embryo_multislice_adaptive_depth_gate.json``; the only accepted
``MultiSliceConfig.correction_mode`` is ``"adaptive_depth_gate"``. The model protocol
is ``two_layer_sum_adaptive_depth_gate_v1``, with ``micro_formula_version=1``.

The downloaded notebook keeps ``RUN_MODE="train"`` as its default. Training
prepares counts and a network, submits an independent tmux job, and passes
that completed job's embeddings to ``sr.analyze``. Optional
``RUN_MODE="provided"`` reads ``EMBEDDING_DIR`` and also runs analysis. Both
create a unique run under ``result/Embryo_multislice/runs/``.

Sections 1–16 show this native pipeline. Its training cell reports the
completed job's epoch, optimizer updates and loss history; the following cells
fit and display that run's domains and UMAPs. The run's
``tutorial_execution.json`` records the actual mode, embedding source,
training execution and analysis configuration. Use that receipt and the
per-run logs to identify the displayed pipeline results.

The current saved example is under
``result/Embryo_multislice/adaptive_depth_gate/``. It contains the final node exports,
complete saved analysis, figures and execution receipt from this workflow.
The notebook's final section reopens saved results without training or fitting.

Inputs and coordinates
-----------------------

The counts contain 7,118 spots and 120,380 shared peaks. Sections
``E11_0-S1``, ``E13_5-S1``, ``E15_5-S1`` and ``E18_5-S1`` contain
1,263, 1,777, 1,949 and 2,129 spots. Input requires nonnegative spot-by-peak
counts, unique spot and genomic peak IDs, section IDs in ``obs["sample_id"]``
(configurable with ``slice_key``), and native XY in ``obsm["spatial"]``.
Filtering retains the union of section-qualified peaks, then applies the
global maximum-frequency threshold. Each shared peak occurs once in the joint
model. Spatial edges stay within sections; there is no cross-section spatial
edge, anchor or alignment loss.

``annotations.tsv`` contains ``spot_id``, ``sample_id`` and ``annotation``.
All 7,118 spots remain in training, fitting and plots, including 81 Unknowns.
Only the 7,037 known annotations enter ARI/NMI scoring. Labels and anatomy
colors are display/evaluation inputs. Set both annotation paths to ``None``
for unannotated data. ``annotation_palette_manuscript.json`` preserves the
reference HEX values, including ``Unknown=#D8DDE3``; its source JSON records
the original color tables and cohort audit.

The additional correction uses edges already in the spatial graph with a unit
displacement
along one coordinate axis and zero displacement along the other (tolerance
1e-6). Four orthogonal sides must all be covered. Eight-sector boundary
coverage and four-side coverage are different statistics. Inspect
the per-section coverage report before training: micrometer coordinates,
rotated grids, hexagonal layouts and irregular points may provide no valid
four-side coverage. Such rows receive zero additional correction; the established
coherence correction remains in effect. The software does not rotate, rescale or replace
these neighbors with kNN automatically.

``DISPLAY_OPTIONS`` transforms coordinate copies for plotting only. Do not
feed these transformed copies back into training. For new data, clear
``spatial_transforms`` and ``slice_titles`` to ``{}``, and set
``spatial_bounds=None`` unless you have selected a suitable display frame.

Receiver correction and per-section budget
------------------------------------------

The model keeps two-layer spot–peak propagation, the sums ``u1+u2`` and
``p1+p2``, and both spatial residuals. Only the first peak-to-spot message is
directly corrected. Let ``h`` be that ungated message, ``g`` the original
null-calibrated receiver gate, and ``S`` the within-section spatial operator.
At epoch 0, the gate uses local deviations of filtered spot–peak degree
(``log2(degree+1)``). After each 50-epoch interval, starting after epoch 50's
loss, it combines the degree deviations with deviations of the current
first-layer molecular messages. Each section uses 1,000 depth-null
permutations to calibrate the degree-only or joint score by tail excess,
with ``g`` in [0,1]. The donor weight always uses the
original ``1-g``:

.. math::

   c_i = \frac{\sum_j S_{ij}(1-g_j)h_j}{\sum_j S_{ij}(1-g_j)},
   \qquad v_i=c_i-h_i.

This ratio describes rows with donor mass above ``epsilon=1e-8``. The
implementation divides the numerator by ``max(donor_mass, epsilon)``.

Eight-sector geometry compares contiguous half-planes and their opposites,
including within-neighbor dispersion, to estimate boundary score ``P``.
Cross-boundary risk ``rho`` compares the actual correction vector ``v`` with
the feature direction separating the two side backgrounds; ``P_eff=P*rho``.

Let ``b_i`` be the neighborhood mean without donor suppression,
``r_i=g_i*(1-P_eff_i)``, ``U_i=1-P_eff_i``, and
``s_i=||b_i-c_i||``. The coherence allowance is:

.. math::

   A_i=[\min(r_i,U_i-r_i)]_+,\qquad
   a_i=\min\left(A_i,\frac{r_i s_i}{\max(\lVert v_i\rVert,\epsilon)}\right).

Invalid coverage or a zero displacement gives zero allowance. ``C`` in [0,1]
measures the donor-weighted directional support for the actual correction.
The coherence-adjusted receiver coefficient is:

.. math::

   \beta_i^{CS}=g_i(1-P_i\rho_i)+a_iC_i.

For four orthogonal sides, row-L2-normalized messages ``z`` give directional
means ``mu_ik`` and within-side dispersion ``sigma_ik``. The outlier score is:

.. math::

   \mu_i=\tfrac14\sum_{k=1}^4\mu_{ik},\qquad
   d_i=\lVert z_i-\mu_i\rVert,

   R_i=\max_k\left(\left\lVert\mu_{ik}
       -\tfrac13\sum_{\ell\ne k}\mu_{i\ell}\right\rVert
       +\sigma_{ik}\right),\qquad
   O_i=\frac{[d_i-R_i]_+}{d_i+R_i}.

Tiny denominators give zero. Let ``u_i`` be the four-side mean of the
**original**, unnormalized messages. Direction evidence is
``D_i=[cos(c_i-h_i,u_i-h_i)]_+`` and ``Q_i=O_i*D_i``. Zero directions or invalid
coverage give zero evidence.

Let ``H_i=[1-P_eff_i-beta_CS_i]_+`` and ``V_i`` indicate technical validity:
four-side coverage, valid features, original coherence coverage, donor denominator
and a nonzero displacement. The local 3% quantity defines a reference budget,
while the actual model distributes it within each section's own refresh state:

.. math::

   e_i^{local}=0.03H_iQ_iV_i,\qquad
   \alpha_s=\frac{\sum_{i\in s}e_i^{local}\lVert v_i\rVert}
                       {\sum_{i\in s}H_iV_i\lVert v_i\rVert},
   \qquad e_i=\alpha_{s(i)}H_iV_i.

A zero denominator gives ``alpha=0`` and ``e=0``; ``0<=alpha_s<=0.03``.
The sums of ``e_i*||v_i||`` and ``e_local_i*||v_i||`` match within each
section. **Q controls the section budget; actual reception is e=alpha*H*V.**
A valid spot with Q=0 can still receive an allocation. No budget crosses
sections.

The receiver coefficient is ``beta=beta_CS+e``. Application preserves the
original float32 order:

.. code-block:: python

   first = h + beta_CS[:, None] * (clean - h)
   result = first + extra[:, None] * (clean - h)

Rows with zero extra retain the coherence-adjusted output directly. On valid rows,
``beta_CS+e<=1-P_eff``. With invalid eight-sector coverage or donor mass, the
receiver falls back to the original `g`; coherence gain and the additional
correction are
zero. The same clamped donor reference is used in ``h + g*(clean-h)``. At
exactly zero donor mass, the nonnegative donor weights give a zero numerator,
so ``clean=0`` and the fallback output is ``(1-g)*h``. For positive donor mass
below epsilon, the denominator remains clamped; no different reference is
substituted. There are no extra learned gate parameters, loss terms,
motif/DNA nodes or thin-structure branch.

Refresh, sampling and optimization
----------------------------------

Gate, protection, coherence and displacement-budget quantities are detached
full-message
caches. They initialize at epoch 0, then refresh every 50 epochs **after the
epoch loss and before checkpoint/export**. For 500 epochs this is 11 states.
Losses at epochs 451–500 use the epoch-450 cache; the final exported embedding
uses the epoch-500 refresh. These full-message computations are recorded
separately from training loss propagations. Budget aggregation uses CPU
float64; application retains float32 operation order.

The T6 profile has seed 1234, 500 epochs, dimension 64, Adam learning
rate 0.001, explicit degree-weighted ego L2 coefficient 0.001, gradient clip
0.5 and spatial weight 0.3. Adam weight decay is zero. Training uses each
section's ``r_by_slice*0.3`` spatial coefficient; full CPU export reads the
already-scaled full spatial adjacency with ``r=1`` so it does not multiply
0.3 twice. There is no scheduler, AMP or early stopping.

Positive shards are 10/10/20/20 for E11/E13/E15/E18. Every epoch performs 20
Adam updates, each combining four section losses with weights 0.25. Independent
seeded shard matching and rotating visits retain coverage. Positives are
resharded every 100 epochs; five negatives per positive are sampled from the
global shared-peak universe, excluding observed pairs, and refresh every five
epochs. This profile configures 10,000 optimizer updates and 40,000 section
loss propagations across 500 epochs.

For new data, ``sr.multislice_training_config`` resolves omitted fields from
filtered positive-edge counts and actual section IDs; each explicit override
wins independently. Joint totals <=2,500 spots use 300 epochs, larger totals
use 500. Each section gets ``max(10, ceil(E/700000))`` shards unless specified.
Do not impose the example's four section names or 10/10/20/20 on other data.

Native stages and durable jobs
-------------------------------

.. code-block:: python

   import seareg as sr

   config = sr.multislice_training_config(counts, device="cpu")
   filtered = sr.filter_multislice_counts(counts, config, "run/filtered")
   network = sr.build_multislice_network(filtered, config, "run/network")
   inputs = sr.prepare_multislice_training(network, config, "run/inputs")
   job = sr.start_multislice_training_job(inputs, config, "run/training")
   training = job.wait()

The immutable network persists the actual graphs consumed by the sampler.
Raw AnnData/H5AD input is also accepted by ``prepare_multislice_training``.
Training uses independent tmux, with command, manifest, stdout/stderr,
epoch diagnostics, atomic checkpoints, recovery script and
``TRAINING_HANDOFF.md`` on disk. The example selects physical GPU1; other
users select an available physical GPU through execution options. This is
an experiment-specific choice, not a package restriction on GPU0.

Rolling state is saved every 10 epochs; full checkpoints and embeddings start
at epoch 300 and repeat every 100, including completion. A 500-epoch tutorial
therefore exports results at 300, 400 and 500. This result-save cadence is
separate from the 50-epoch gate refresh and from scientific parameters.
Exact native recovery restores model and all gate,
geometry, protection, coherence and displacement-budget caches, Adam, RNG,
sampling/data order
and shard matching, with matching scientific and numerical identities. It does
not refresh caches from current features during load. Recovery requires the
complete compatible training run. Embedding exports are analysis inputs,
not full-state recovery checkpoints.

Saved results and analysis
--------------------------

``result/Embryo_multislice/adaptive_depth_gate/500_epoch/`` contains the final
spot/peak node exports and their ``export.json`` metadata. The parent result
directory contains the complete native ``analysis/``, ``figures/``, display
colors and ``tutorial_execution.json``. The receipt identifies the inputs,
training configuration, actual embedding source and downstream analysis.
Checkpoints are retained in the complete training run and are not distributed.

Analysis uses float64 row-L2, PCA20 ``covariance_eigh`` (seed1234), an auto
15-neighbor Euclidean graph and directed, unweighted Louvain (seed0).
The profile sets ``target_domains=[16]`` and ``fixed_resolution=None`` for
``sr.analyze`` exact-K search in [0,3], with at most 20 iterations. Tissue
annotations do not select the partition. K is a downstream analysis target,
not a training head. The saved analysis retains the actual search trace and
native labels from that execution.

Both ``train`` and ``provided`` use this analysis. Failure to reach the target
is reported. Choose an appropriate target or explicit resolution for new data.

The selected example palette is bound to its exact cohort and partition membership.
Aliases and colors affect display copies only. Other partitions use annotation
matching or default colors. Native labels remain available.

Figures and reopening
---------------------

T6 exports reference maps, spatial domains, and spot/joint UMAP rows titled
``Sections``, ``Spatial domains`` and ``Reference annotations``. Each row uses
one saved coordinate table; all shared peaks remain gray in joint panels.
Figures are editable vector PDF/SVG plus high-resolution PNG, without
horizontal gridlines through data. Display or reopen actions reuse saved
labels and coordinates without clustering, UMAP fitting or training.

To reopen the current saved analysis from the repository root:

.. code-block:: python

   from seareg.analysis import load_result

   saved = load_result("result/Embryo_multislice/adaptive_depth_gate/analysis")
   domains = saved.domains()
   spot_umap = saved.spot_embedding()
   joint_umap = saved.joint_embedding()

These calls verify the saved artifact and load its existing labels, graph and
coordinates without fitting. Keep the complete ``analysis/`` directory intact
so its registered file hashes can be verified.

``tutorial_execution.json`` identifies the actual embedding source, training
execution, analysis configuration, model/formula, cohort and annotation
provenance. In provided mode, the receipt sets ``training_executed=false`` and
``training_run=null`` while retaining the configured ``training`` and
``execution`` dictionaries. That action loads an export rather than training
a model. See :doc:`workflow` for durable jobs and full-state recovery.
