# Tutorial input data

The complete GitHub repository includes five processed count/motif input pairs and a four-section embryo count matrix as lossless gzip archives. `git clone` and **Download ZIP** include their actual bytes. Wheel and source distributions omit these data and the notebooks. Complete model checkpoints are not distributed.

```text
data/<dataset>/
  counts.h5ad.gz                 # P22_ATAC uses numbered .part* files
  motifs.h5.gz
```

| Dataset | Input directory |
| --- | --- |
| Mouse embryo (E13, spatial ATAC–RNA-seq) | `ME13_joint_2187/` |
| Mouse embryo (E13, spatial-ATAC-seq) | `ME13_ATAC_2275/` |
| Mouse brain (P22, spatial ATAC–RNA-seq) | `P22_ATAC/` |
| Mouse brain (P22, spatial CUT&Tag–RNA-seq, H3K27ac) | `P22_H3K27ac/` |
| Human tonsil (spatial-ATAC-seq) | `Tonsil/` |
| Developing mouse brain (E11.0–E18.5, MISAR-seq) | `Embryo_multislice/` |

The developing mouse brain dataset (`Embryo_multislice`) contains 7,118 spots × 120,380 shared peaks across `E11_0-S1`, `E13_5-S1`, `E15_5-S1` and `E18_5-S1`. It has counts only, without motifs. `annotations.tsv` contains `spot_id`, `sample_id` and tissue `annotation`; 7,037 spots have known labels and 81 are `Unknown`. T6 shows these annotations immediately after reading counts. They are separate display/evaluation inputs and do not enter model training or domain identification.

T6 uses [`annotation_palette_manuscript.json`](Embryo_multislice/annotation_palette_manuscript.json) for reference anatomy colors, including `Unknown=#D8DDE3`. The CSV/TSV tables and [`annotation_palette_manuscript_source.json`](Embryo_multislice/annotation_palette_manuscript_source.json) record the source colors and spot identities.

The notebooks prepare the selected counts file automatically. T1–T5 also unpack selected bundled motifs in training mode; motif import or scanning occurs after peak filtering. For CLI/API use, extract all declared inputs from the repository root:

```bash
python tools/prepare_data.py --dataset all
```

[manifest.json](manifest.json) records compressed-part and original-file identities, sizes and shapes. Archive extraction joins split gzip files, validates their bytes, and creates the original `counts.h5ad` and `motifs.h5` without replacing existing files. Unpacked files are ignored by Git. Extraction performs no preprocessing or model computation. The CLI above checks the declared example identities.

The notebook interface accepts any existing `ATAC_PATH` and optional `MOTIF_PATH`, including paths outside this repository. A missing path that exactly identifies a bundled input triggers archive extraction. Missing counts raise an error; missing motifs follow `MOTIF_OPTIONS` for reading or scanning. Matrix values, identifiers and peak alignment are validated by the workflow.

All six notebooks default to `RUN_MODE="train"`, preparing inputs and training a model before analyzing its final embeddings. To use existing embeddings, set `RUN_MODE="provided"` and choose `EMBEDDING_DIR`; T6's configured provided source is `result/Embryo_multislice/cs_matched03/500_epoch/`. Matching counts supply identities, section IDs and native spatial coordinates. Both modes write analysis and figures in a unique run under `OUTPUT_DIR`. Reopen T6's saved labels and UMAPs with `seareg.analysis.load_result("result/Embryo_multislice/cs_matched03/analysis")` without refitting. See [the tutorial guide](../tutorials/README.md).

Original studies: [Zhang et al., Nature (2023)](https://doi.org/10.1038/s41586-023-05795-1), GEO GSE205055 (T1/T3/T4); [Deng et al., Nature (2022)](https://doi.org/10.1038/s41586-022-05094-1), GEO GSE171943 (T2/T5); [Jiang et al., Nature Methods (2023)](https://doi.org/10.1038/s41592-023-01884-1), NODE OEP003285 (T6). Inputs start at processed chromatin count matrices with spatial coordinates and peak–motif annotations. RNA, FASTQ processing and peak calling are outside this tutorial input boundary.
