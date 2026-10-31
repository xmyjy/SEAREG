# Tutorials

T1–T5 cover single-section training with motifs. T6 adds joint training and analysis of four developing mouse brain sections, with reference tissue annotations, spatial domains and three-panel spot/joint UMAP comparisons. See the [method overview](../docs/unified_framework.rst) for how the shared representations support further regulatory-feature analyses.

All notebooks expose paths and parameters in the first code cell. T1–T5 use `Workflow`; T6 uses the dedicated `MultiSliceConfig` and the same durable job/result interfaces. Use a complete GitHub checkout or Download ZIP: wheel and source distributions omit the notebooks, data and provided results. Install the default Python 3.10.22 / PyTorch 2.2.0 / CUDA 12.1 environment following [installation](../docs/installation.rst); that guide also provides an explicit CPU alternative.

All six notebooks default to `RUN_MODE="train"` to prepare the selected data, train a model and analyze its final embeddings. Configure the first code cell and run the cells in order. Install tmux (`conda install -c conda-forge tmux` on Linux) and select CPU or CUDA; the notebook submits an independent job and waits for its completed result before analysis. Use `provided` mode to analyze existing embeddings.

The repository includes these default inputs and named embeddings:

| Notebook | Dataset | Spots | Profile epochs | Domain analysis |
| --- | --- | ---: | ---: | ---: |
| [T1](T1_Mouse_Embryo_E13_Spatial_ATAC_RNA.ipynb) | Mouse embryo (E13, spatial ATAC–RNA-seq) | 2,187 | 300 | 13 |
| [T2](T2_Mouse_Embryo_E13_Spatial_ATAC.ipynb) | Mouse embryo (E13, spatial-ATAC-seq) | 2,275 | 300 | 17 |
| [T3](T3_Mouse_Brain_P22_Spatial_ATAC_RNA.ipynb) | Mouse brain (P22, spatial ATAC–RNA-seq) | 9,215 | 500 | 13 |
| [T4](T4_Mouse_Brain_P22_Spatial_CUTTag_RNA_H3K27ac.ipynb) | Mouse brain (P22, spatial CUT&Tag–RNA-seq, H3K27ac) | 9,370 | 500 | 11 |
| [T5](T5_Human_Tonsil_Spatial_ATAC.ipynb) | Human tonsil (spatial-ATAC-seq) | 2,488 | 500 | 3 |
| [T6](T6_Mouse_Brain_Development_MISAR_seq.ipynb) | Developing mouse brain (E11.0–E18.5, MISAR-seq) | 7,118 | 500 | Target K16 |

T1 additionally demonstrates `sr.spatial_scores` for one peak and one motif.
It retains raw cosine associations and displays whole-section standardized
MSAS/PSAS with square spots and a shared viridis z-score scale. A third panel
shows the same peak's observed accessibility as discrete gray 0 / red 1 blocks.
It exports `feature_spatial_scores.tsv`, `peak_accessibility.tsv` (raw counts
and detection values), and editable PNG/PDF/SVG figures. Set
`SCORE_PEAK` and `SCORE_MOTIF` to native embedding IDs to choose your features.
The default selects the Pou3f2-associated peak and PRDM1 motif when present;
otherwise it reports the first available feature. Counts-only runs show the
peak score and observed accessibility without a motif panel.

## Joint developmental sections (T6)

T6 uses `counts.h5ad` with section identities in `obs["sample_id"]`, a shared genomic peak axis and native spatial coordinates. Its profile is [`configs/Embryo_multislice_adaptive_depth_gate.json`](configs/Embryo_multislice_adaptive_depth_gate.json). For your own sections, set the input/output paths and name every section in `TRAINING_OPTIONS["slice_shards"]`. Set `ANNOTATIONS_PATH = None` and `ANNOTATION_PALETTE_PATH = None` when reference labels are unavailable.

The example also defines tissue-specific display transforms, titles and axis bounds. For new data, add the following after `DISPLAY_OPTIONS` is created in the first code cell, before running subsequent cells:

```python
DISPLAY_OPTIONS["spatial_transforms"] = {}
DISPLAY_OPTIONS["spatial_bounds"] = None
DISPLAY_OPTIONS["slice_titles"] = {}
```

This selects native XY with automatic bounds. Add transforms and titles only when they suit your coordinate system, even if your section IDs match the examples. Display options act on coordinate copies.

The profile and `MultiSliceConfig` use `correction_mode="adaptive_depth_gate"` (`two_layer_sum_adaptive_depth_gate_v1`, micro formula 1). Donor weights are `1-g`; reception combines `g*(1-P*rho)+a*C` and `e=alpha_s*H*V`. `Q=O*D` sets each section’s matched local-3% budget; it is not a pointwise 3% multiplier. The profile uses 500 epochs, 100/5 reshard/negative refresh, 10/10/20/20 shards and 20 joint updates/epoch. Detached full-message caches refresh every 50 epochs after the epoch loss. Rolling recovery state is saved every 10 epochs; full checkpoints and embeddings start at 300 and repeat every 100 epochs, including completion. The 500-epoch example exports results at 300, 400 and 500. Train/provided analyses call `sr.analyze` with `target_domains=[16]` and `fixed_resolution=None`. See the [multislice guide](../docs/multislice.rst) for the correction, sampling and optimizer protocol.

Before training, inspect `seareg.multislice.geometry.spatial_coverage_report(coordinates, spatial, slice_codes)`: the additional correction requires unit, axis-aligned edges on all four orthogonal sides (tolerance 1e-6), in the original within-section graph. The report separates eight-sector and four-side coverage by section. Rotated, micrometer-scale, hexagonal or irregular coordinates may have no valid coverage, making the extra term zero. No automatic rotation, scaling or kNN substitution is performed. Geometric preflight cannot report alpha before messages have been computed; runtime diagnostics record actual alpha and budgets. Display coordinate copies never enter training. For new data, `sr.multislice_training_config` resolves omitted fields independently from filtered edge counts; explicit per-field values win. Do not copy the reference section names/shards blindly.

Reference annotations appear immediately after reading counts. The tutorial exports reference maps, spatial domains, and spot/joint UMAP rows titled `Sections`, `Spatial domains` and `Reference annotations`. Each row shares saved coordinates; shared peaks are gray in joint panels. Selected-example domain colors apply only to its exact cohort and partition. Other results use annotation matching or default colors.

## Configure the inputs and parameters

The first code cell is the common configuration interface. Each notebook's JSON profile supplies its example defaults. `DATASET` is a run label; edit the paths and parameters for your data. The fields below describe T1–T5; T6's section and annotation options are described above.

For your own data, edit the existing assignments in that cell:

```python
DATASET = "my_section"
ATAC_PATH = Path("/data/my_section/counts.h5ad")
MOTIF_PATH = Path("/data/my_section/motifs.h5")
OUTPUT_DIR = Path("/results/my_section")
TARGET_K = 8
RUN_MODE = "train"
```

Inputs and outputs may be outside the repository. Use absolute paths when working there. The bundled defaults resolve from the repository root, including when Jupyter starts inside `tutorials/`. After changing the configuration, rerun the cells in order so all stages use the selected inputs.

`FILTERING_OPTIONS`, `NETWORK_OPTIONS`, `TRAINING_OPTIONS`, `ANALYSIS_OPTIONS` and `EXECUTION_OPTIONS` expose the profile parameters. Edit these dictionaries before the cell constructs the configuration objects. For example, CPU training uses:

```python
TRAINING_OPTIONS["device"] = "cpu"
EXECUTION_OPTIONS["physical_gpu"] = None
```

Choose `TARGET_K` for your data; the example domain counts are defaults. The remaining analysis options include PCA dimensions, neighbor count and UMAP settings. Set `ANALYSIS_OPTIONS["domains"]`, `["spot_umap"]` and `["joint"]` independently to select the analyses and figures. When domains are disabled, UMAP plots use the available node types without domain labels. Disabling all three skips domain and UMAP outputs. T1's feature-score cell and T6's reference-map cell are separate analyses/displays.

`MOTIF_OPTIONS` controls motif preparation:

| Desired behavior | Configuration |
| --- | --- |
| Read your existing peak–motif matrix | Set `MOTIF_PATH`; use `enabled=True` and `source="read"` to require it. |
| Read a matrix when available, otherwise scan | Use `enabled=True`, `source="auto"`; declare the actual `genome`, and optionally a matching local `fasta`. |
| Scan even when a matrix exists | Use `enabled=True`, `source="scan"` and the reference settings. |
| Train without motif associations | Set `MOTIF_OPTIONS["enabled"] = False`; `MOTIF_PATH` can be `None`. |

New scans use the existing R/JASPAR2024 backend after peak filtering. The visible default `species_names` is `['Homo sapiens', 'Mus musculus']`; select the species appropriate for your input. `rscript`, `threads` and `download_reference` are also configurable. `SEAREG_REFERENCE_ASSEMBLY`, `SEAREG_REFERENCE_FASTA`, `SEAREG_MOTIF_SPECIES` and `SEAREG_RSCRIPT` can supply initial values. Existing annotations need no R or reference resources. See [motif annotation](../docs/motif_annotation.rst).

## Input preparation and output locations

Use the selected count matrix and optional motif file. Bundled compressed examples are extracted when needed; motif import or scanning follows peak filtering. See [input schema](../docs/input_data.rst) for your own data.

Each execution creates a unique directory under the chosen output base:

```text
OUTPUT_DIR/
  runs/<mode>_<time>_<id>/
    artifacts/<stage>/<key>/     # filtered counts, selected motifs, network, training inputs
    workflow.json               # prepared stage references in training mode
    training_job/               # tmux request, logs, handoff, recovery script
    training/                   # initial attempt's model and embeddings
      embeddings/<epoch>_epoch/
    analysis/
    figures/
    tutorial_execution.json
```

The training-specific entries are created only in `train` mode. T6 uses separate immutable `filtered/`, `network/` and `training_inputs/` stages; its sampler consumes the network's saved graph values. A recovered attempt can have a different training path; the receipt records the completed path. The default `OUTPUT_DIR` is `ROOT / "result" / DATASET`. T6's current saved result is under `result/Embryo_multislice/adaptive_depth_gate/`, with its final `500_epoch/` exports, complete `analysis/`, figures and execution receipt. T1–T5 bundled exports are under `result/<example>/training/embeddings/<epoch>_epoch/`. New training reads and keeps its exports in its own run. To reanalyze one later, select that run's export directory as `EMBEDDING_DIR`.

## Training and recovery

Set `SEAREG_PHYSICAL_GPU` before launching Jupyter to select a physical GPU; T1–T5 profiles use card 0, and the T6 example uses card 1. Check the selected physical card before launch; this example choice does not restrict other users’ GPU selection. One progress bar, final training statistics and a loss table appear in the training cell. Complete runtime logs remain in the run directory. The training cell submits `sr.start_training_job(...)` (T1–T5) or `sr.start_multislice_training_job(...)` (T6) and waits. Closing Jupyter or SSH does not terminate the job. `training_job/TRAINING_HANDOFF.md` records the session, runner/worker PIDs, GPU, parameters, epoch, latest checkpoint and log paths.

Reconnect to the same job with:

```python
job = sr.load_training_job("/results/my_section/runs/<run>/training_job")
training = job.wait()
```

Recover an interrupted attempt using its generated script:

```bash
bash /results/my_section/runs/<run>/training_job/resume.sh
# Append a specific complete checkpoint path when needed.
```

The full recovery checkpoint `checkpoints/latest.pt` is saved atomically every 10 epochs and at each result export. All tutorial profiles start numbered checkpoints and named embeddings at epoch 300, repeat every 100 epochs and also save at completion (`export_start_epoch=300`, `export_every=100`). A 500-epoch example exports results at 300, 400 and 500; shorter runs still save their final result. T6's 50-epoch gate refresh is independent of result saving. Recovery restores model, Adam, RNG and sampler states together, including saved gate, geometry, protection, coherence and displacement-budget caches. Resume with the same inputs, package and numerical settings. Each attempt retains its own outputs, and reconnecting to a completed job returns its results. See [workflow and recovery](../docs/workflow.rst).

## Analyze existing embeddings

To reanalyze existing outputs instead of training, set both source variables in the first cell:

```python
RUN_MODE = "provided"
EMBEDDING_DIR = Path("/results/previous_run/training/embeddings/300_epoch")
```

Also set `ATAC_PATH` to the corresponding counts. The directory must contain `export.json` metadata and its propagated spot/peak matrices, plus motif embeddings when present. T6's configured `EMBEDDING_DIR` points to its current adaptive_depth_gate `500_epoch/` export when this optional mode is chosen. A bundled example and your own saved run use the same loader. It checks the selected propagated files, matrix axes and values against the export metadata and counts. This mode skips training preparation and model training; no motif file, R, GPU, tmux or complete checkpoint is required.

Both modes pass their selected `embeddings` to the same analysis and plotting cells. New spots or peaks generally require training a model for that data; existing embeddings and recovery checkpoints describe the nodes of their original run. The execution record identifies the actual embedding source. In provided mode, T6 records `training_executed=false` and `training_run=null`, while retaining its configured `training` and `execution` dictionaries.

To read T6's current saved labels and UMAP coordinates without fitting, reopen `result/Embryo_multislice/adaptive_depth_gate/analysis/` with `seareg.analysis.load_result`. The notebook's final section demonstrates reopening a saved analysis. This reads the same result that supplied the published figures; selecting `provided` mode instead computes a new analysis from the selected embeddings.

## Analysis, figures and execution records

All profiles use float64 row-L2, PCA20 and Scanpy 15 Euclidean neighbors. T1–T5 use exact-K Louvain search; T6’s train/provided pipeline also uses exact-K search (target 16, no fixed resolution), PCA20 `covariance_eigh`/seed1234, auto15 neighbors and Louvain seed0. Spot UMAP uses the spot graph. Joint UMAP normalizes each available node type and fits all selected nodes directly. Both use seed 1234 and `min_dist=0.08`. T1–T5 show native spatial XY; T6 applies the reference display orientations to copies of the coordinates.

The selected figures use the domain labels computed in the current execution when domain identification is enabled. PNG, editable vector PDF and SVG go to its `figures/` directory. Display settings remain in the plotting cells.

P22 ATAC, P22 H3K27ac and tonsil retain their established example names and colors in all three panels. The small tables in `styles/` associate spot IDs with these display names. The tutorial applies them only when the computed partition matches exactly; new cohorts or different partitions keep their native labels and default colors. This display mapping leaves clustering and coordinates unchanged. Pass `palette={label: color, ...}` to a plotting function to choose colors explicitly.

`tutorial_execution.json` links the input data, selected model embeddings, configurations and figure paths. Numerical analysis details are saved in `analysis/analysis_manifest.json`.
