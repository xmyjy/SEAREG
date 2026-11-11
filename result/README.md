# Provided embeddings and new runs

The complete GitHub checkout and Download ZIP include named node exports and their `export.json` metadata. Wheel and source distributions omit these results. Complete model checkpoints are not distributed.

| Export | Location |
| --- | --- |
| T1–T2 spot, peak and motif embeddings | `<dataset>/training/embeddings/300_epoch/` |
| T3–T5 spot, peak and motif embeddings | `<dataset>/training/embeddings/500_epoch/` |
| T6 adaptive_depth_gate spot and peak embeddings | [`Embryo_multislice/adaptive_depth_gate/500_epoch/`](Embryo_multislice/adaptive_depth_gate/README.md) |

All six notebooks default to `RUN_MODE="train"` and analyze their completed training job's final exports. Optional `RUN_MODE="provided"` reads `EMBEDDING_DIR` with matching `ATAC_PATH` and computes a new analysis. The loader validates propagated matrices and node axes against export metadata and counts.

T6 publishes one current adaptive_depth_gate result under [`Embryo_multislice/adaptive_depth_gate/`](Embryo_multislice/adaptive_depth_gate/README.md): final `500_epoch/` node exports, the complete saved `analysis/`, figures, display colors and `tutorial_execution.json`. The receipt identifies the inputs, training and downstream analysis used by the notebook. Use `seareg.analysis.load_result` to reopen its saved labels, graph and UMAP coordinates without fitting. Node exports support analysis; full-state recovery checkpoints remain in the training run and are not distributed.

Each pipeline execution creates `OUTPUT_DIR/runs/<mode>_<time>_<id>/`. Models and embeddings stay in the completed attempt's training directory; analysis tables, figures and `tutorial_execution.json` stay with that run. New outputs are ignored by Git. The default base is `result/<dataset>/`; input/output locations may be elsewhere.

See the [tutorial guide](../tutorials/README.md) for source selection, output layout and training recovery.
