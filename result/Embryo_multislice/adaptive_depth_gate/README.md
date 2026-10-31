# SEAREG adaptive_depth_gate: developing mouse brain

This is the current result of Tutorial 6's complete native training and downstream analysis on four MISAR-seq sections. It contains 7,118 spots, including 81 Unknown tissue annotations, and 120,380 shared peaks. Tissue annotations support the reference maps and post-fit display colors; they are not training inputs.

SEAREG 1.0.0 trains adaptive_depth_gate for 500 epochs with seed 1234, 64 dimensions, 10/10/20/20 section shards, 20 Adam updates and 80 section loss propagations per epoch. Sampling refreshes use 100/5, model gates refresh every 50 epochs, and rolling checkpoints use 10 epochs. Numbered result exports occur at epochs 300, 400 and 500. TRAINING_SUMMARY.json records the actual 10,000 updates, 40,000 propagations and 11 gate states.

## Files

- 500_epoch/: all five original final-export files, including propagated and ego spot/peak matrices and export.json.
- analysis/: the complete, sealed native analysis with labels, graph, PCA, saved spot/joint UMAPs and its integrity manifests.
- figures/: reference annotations, spatial domains, spot UMAP and joint UMAP; each has PNG, editable PDF and editable SVG.
- tutorial_execution.json: the actual input, training, analysis and display receipt.
- domain_annotation_colors.tsv, display_palette.json and approved_domain_palette.json: display mappings and the membership-checked example palette.
- spatial_coverage.json: per-section native coordinate/graph coverage.
- training_evidence/: the actual configuration, method, epoch diagnostics, final status, job request and original_training_manifest.json.
- bundle_manifest.json: file hashes and original source mappings.

## Load the final embeddings

From the repository root in the tutorial environment:

```python
from pathlib import Path
import sys
ROOT = Path.cwd()
sys.path.insert(0, str(ROOT / "tutorials"))
from seareg_tutorial_data import prepare_input_file
from seareg_tutorial_embeddings import load_tutorial_embeddings
BUNDLE = ROOT / "result/Embryo_multislice/adaptive_depth_gate"
counts = prepare_input_file(ROOT / "data/Embryo_multislice/counts.h5ad", bundle_root=ROOT)
embeddings = load_tutorial_embeddings(BUNDLE / "500_epoch", counts_path=counts)
print(embeddings.spot.shape, embeddings.peak.shape)
```

## Reopen the saved analysis

```python
from seareg.analysis import load_result
saved = load_result(BUNDLE / "analysis")
domains = saved.domains()
spots = saved.spot_embedding()
joint = saved.joint_embedding()
print(domains.observed_k, len(domains.labels), len(spots.table), len(joint.table))
```

Reopening reads the saved labels and coordinates without fitting or training. Optional Tutorial 6 RUN_MODE="provided" instead selects final embeddings for a new downstream analysis in a new run directory. To resume your own training, use its complete training run and full-state checkpoint. This bundle includes final embeddings and saved analysis; recovery checkpoints are not included.
