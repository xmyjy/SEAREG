# Changes

## 1.0.0 — 2026-10-31

- First public release of SEAREG, with six tutorials, an English documentation website, installation and data guides, and Python API/CLI references.
- Provide single-section spot–peak–motif graph learning and motif-free `adaptive_depth_gate` joint-section learning to reduce the influence of spots with locally anomalous sequencing depth, with risk-modulated boundary protection, donor coherence and a per-section 3% reference displacement budget.
- Expose staged count filtering, motif annotation, saved networks, training inputs, durable tmux training jobs, native analysis and result reopening. Retain model, Adam, random states, sampler order and multislice gate caches for full-state recovery.
- Save rolling recovery state every 10 epochs; save full checkpoints and node embeddings from epoch 300 every 100 epochs and at completion. A 500-epoch tutorial exports results at 300, 400 and 500. Multislice gate caches refresh every 50 epochs independently of result saving.
- Include tutorial inputs, final named embeddings, saved domains and UMAPs, editable PDF/SVG figures, PNG previews and execution receipts. Default to fresh training; optional provided mode analyzes selected exports, while saved-analysis loading reopens existing labels and coordinates without fitting.
- Use an inclusive 2,500-spot boundary for single-section peak filtering and automatic epoch defaults. Resolve new-data shard counts from retained positive edges, with explicit field overrides taking precedence.
- Supply the pinned Linux Python 3.10.22 / PyTorch 2.2.0 / CUDA 12.1 tutorial environment and a CPU alternative. Publish software and citation metadata as version 1.0.0.
