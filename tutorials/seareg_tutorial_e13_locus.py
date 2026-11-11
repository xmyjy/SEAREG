"""Compute and plot the E13 Pou3f2 locus from fragment, sequence and RNA inputs."""
import gzip
import sqlite3

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.font_manager import FontProperties
from matplotlib.patches import ConnectionPatch, PathPatch, Rectangle
from matplotlib.path import Path as DrawingPath
from matplotlib.textpath import TextPath
from matplotlib.transforms import Affine2D
import numpy as np
import pandas as pd

from seareg.plotting import DOMAIN_COLORS


CHROM = "chr4"
WINDOW = (22_475_000, 22_505_500)
PEAK = (22_497_217, 22_498_484)
GENE = "Pou3f2"
DOMAINS = ("D11", "D1", "D6", "D7", "D9")
INPUT_FILES = {
    "fragments": "GSM6801813_ME13_50um_fragments.tsv.gz",
    "fragments_index": "GSM6801813_ME13_50um_fragments.tsv.gz.tbi",
    "rna": "GSM6799937_ME13_50um_matrix_merge.tsv.gz",
    "fasta": "mm10.fa",
    "fasta_index": "mm10.fa.fai",
    "gtf": "gencode.vM25.annotation.gtf.gz",
    "jaspar": "JASPAR2024.sqlite",
    "archr": "ME13_p2g_cor0.2_FDR0.1_annotated.tsv",
}


def fragment_tracks(fragments_path, spot_domains, depth_csv):
    """Mean 50-bp full-fragment CP10K over all spots in each displayed domain."""
    import pysam

    spot_domains = spot_domains.loc[spot_domains.isin(DOMAINS)]
    if depth_csv.is_file():
        depth = pd.read_csv(depth_csv).set_index("barcode").total_fragments.to_dict()
    else:
        depth = dict.fromkeys(spot_domains.index, 0)
        with gzip.open(fragments_path, "rt") as fragments:
            for line in fragments:
                if line.startswith("#"):
                    continue
                _, _, _, spot, multiplicity, *_ = line.rstrip("\n").split("\t")
                if spot in depth:
                    depth[spot] += int(multiplicity)
    start, end = WINDOW
    bins = np.arange(start, end + 1, 50)
    coverage = np.zeros((len(spot_domains), len(bins) - 1))
    row_by_spot = {spot: index for index, spot in enumerate(spot_domains.index)}
    with pysam.TabixFile(str(fragments_path)) as fragments:
        for line in fragments.fetch(CHROM, start, end):
            _, left, right, spot, multiplicity, *_ = line.split("\t")
            if spot not in row_by_spot or depth[spot] <= 0:
                continue
            first = (max(int(left), start) - start) // 50
            stop = (min(int(right), end) - start + 49) // 50
            coverage[row_by_spot[spot], first:stop] += int(multiplicity) * 10_000 / depth[spot]
    tracks = pd.DataFrame({"position": bins[:-1] + 25, "bin_start": bins[:-1], "bin_end": bins[1:]})
    for domain in DOMAINS:
        tracks[f"{domain}_mean_fragment_cp10k"] = coverage[spot_domains.to_numpy() == domain].mean(axis=0)
    return tracks


def read_pou3f2_rna(rna_path, spot_domains):
    """Align GEO raw counts by paired spot barcode and compute log1p."""
    with gzip.open(rna_path, "rt") as matrix:
        barcodes = next(matrix).rstrip("\n").split("\t")
        counts = next(line.rstrip("\n").split("\t")[1:]
                      for line in matrix if line.startswith(f"{GENE}\t"))
    counts = pd.Series(np.asarray(counts, dtype=np.int32),
                       index=[f"{barcode}-1" for barcode in barcodes])
    rna = spot_domains.loc[spot_domains.isin(DOMAINS)].rename("domain").to_frame()
    rna["raw_count"] = counts.loc[rna.index]
    rna["Pou3f2_log1p_count"] = np.log1p(rna.raw_count.to_numpy(dtype=np.float32))
    rna.index.name = "spot_id"
    return rna.reset_index()


def scan_prdm1(jaspar_path, fasta_path):
    """Calculate a PSSM from JASPAR MA0508.4 and scan both strands of Peak1."""
    from Bio import motifs
    from Bio.Seq import Seq
    import pysam

    with sqlite3.connect(jaspar_path.resolve().as_uri() + "?mode=ro", uri=True) as db:
        matrix = pd.read_sql_query(
            """SELECT MATRIX_DATA.row AS base, MATRIX_DATA.col AS position, MATRIX_DATA.val AS count
               FROM MATRIX_DATA JOIN MATRIX USING (ID)
               WHERE BASE_ID = 'MA0508' AND VERSION = 4""", db)
    pfm = matrix.pivot(index="base", columns="position", values="count")
    motif = motifs.Motif(counts={base: pfm.loc[base].to_list() for base in "ACGT"})
    motif.pseudocounts = 0.8
    pssm = motif.pssm
    threshold = pssm.distribution(background=dict.fromkeys("ACGT", 0.25),
                                  precision=10_000).threshold_fpr(1e-5)
    with pysam.FastaFile(str(fasta_path)) as fasta:
        sequence = Seq(fasta.fetch(CHROM, *PEAK).upper())

    hits = []
    for offset, score in pssm.search(sequence, threshold=threshold, both=True):
        start = int(offset) % len(sequence)
        end = start + len(motif)
        matched = sequence[start:end]
        if offset < 0:
            matched = matched.reverse_complement()
        hits.append((CHROM, PEAK[0] + start, PEAK[0] + end,
                     "+" if offset >= 0 else "-", float(score), threshold, str(matched)))
    return pfm, pd.DataFrame(hits, columns=[
        "chrom", "start", "end", "strand", "pssm_score", "threshold", "matched_sequence"])


def read_local_links(archr_path):
    links = pd.read_csv(archr_path, sep="\t")
    local = links.loc[
        links.symbol.eq(GENE) & links.peak_chr.eq(CHROM)
        & links.peak_end.gt(WINDOW[0]) & links.peak_start.lt(WINDOW[1]),
        ["peak_name", "peak_start", "peak_end", "symbol", "Correlation", "FDR"],
    ].sort_values("peak_start")
    local["is_selected"] = local.peak_start.eq(PEAK[0]) & local.peak_end.eq(PEAK[1])
    return local


def read_gene_model(gtf_path):
    rows = []
    with gzip.open(gtf_path, "rt") as gtf:
        for line in gtf:
            if f'gene_name "{GENE}"' not in line:
                continue
            chrom, _, feature, start, end, _, strand, _, _ = line.rstrip("\n").split("\t")
            if chrom == CHROM and feature in {"gene", "exon", "CDS"}:
                rows.append((feature, int(start) - 1, int(end), strand))
    return pd.DataFrame(rows, columns=["feature", "start", "end", "strand"]).drop_duplicates()


def draw_prdm1_logo(ax, pfm):
    probabilities = pfm.div(pfm.sum()).clip(lower=1e-12)
    information = 2 + (probabilities * np.log2(probabilities)).sum()
    heights = probabilities.mul(information)
    colors = {"A": "#33A02C", "C": "#1F78B4", "G": "#FFB000", "T": "#E31A1C"}
    font = FontProperties(family="DejaVu Sans", weight="bold")
    for i, column in enumerate(heights.columns, start=1):
        bottom = 0
        for base, height in heights[column].sort_values().items():
            if height <= 0.015:
                continue
            glyph = TextPath((0, 0), base, size=1, prop=font)
            box = glyph.get_extents()
            shape = (Affine2D().translate(-box.x0, -box.y0).scale(0.82 / box.width, height / box.height)
                     .translate(i - 0.41, bottom))
            ax.add_patch(PathPatch(glyph, transform=shape + ax.transData, color=colors[base], lw=0))
            bottom += height
    ax.set(xlim=(0.5, len(pfm.columns) + 0.5), ylim=(0, 2),
           xticks=range(1, len(pfm.columns) + 1), yticks=[0, 1, 2], ylabel="Bits")
    ax.set_title("PRDM1 motif", pad=5)
    ax.tick_params(length=2, pad=2, labelsize=7)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(False)


def plot_locus(tracks, links, pfm, hits, gene_model, rna, peak_ids):
    """Draw the locus with a shared mm10 axis and aligned RNA distributions."""
    fig = plt.figure(figsize=(9.4, 5.5))
    genome_x = (0.087, 0.69)
    arc = fig.add_axes([genome_x[0], 0.77, genome_x[1], 0.12])
    peaks = fig.add_axes([genome_x[0], 0.72, genome_x[1], 0.043], sharex=arc)
    tracks_y = [0.62, 0.52, 0.42, 0.32, 0.22]
    coverage = [fig.add_axes([genome_x[0], y, genome_x[1], 0.092], sharex=arc)
                for y in tracks_y]
    gene = fig.add_axes([genome_x[0], 0.142, genome_x[1], 0.064], sharex=arc)
    logo = fig.add_axes([0.66, 0.77, 0.115, 0.12])
    expression = fig.add_axes([0.81, 0.213, 0.17, 0.506])

    colors = mpl.colormaps["Reds"]
    norm = Normalize(0, 3, clip=True)
    gene_span = gene_model.loc[gene_model.feature.eq("gene")].iloc[0]
    tss = gene_span.end if gene_span.strand == "-" else gene_span.start
    for link in links.itertuples(index=False):
        midpoint = (link.peak_start + link.peak_end) / 2
        height = 0.10 + 0.73 * abs(midpoint - tss) / (WINDOW[1] - WINDOW[0])
        path = DrawingPath([(midpoint, 0), ((midpoint + tss) / 2, height), (tss, 0)],
                           [DrawingPath.MOVETO, DrawingPath.CURVE3, DrawingPath.CURVE3])
        arc.add_patch(PathPatch(path, fill=False, color=colors(norm(-np.log10(link.FDR))), lw=1.1, alpha=0.75))
    arc.set(xlim=WINDOW, ylim=(0, 0.30))
    for link in links.loc[links.is_selected].itertuples(index=False):
        fig.text(0.5, 0.967, f"Selected Peak2Gene link: r = {link.Correlation:.2f}, FDR = {link.FDR:.2e}",
                 ha="center")
    colorbar = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=colors),
                           cax=fig.add_axes([0.234, 0.883, 0.10, 0.019]), orientation="horizontal")
    colorbar.set_ticks([0, 3])
    colorbar.ax.xaxis.set_ticks_position("top")
    colorbar.ax.tick_params(length=0, pad=2, labelsize=8)
    colorbar.solids.set_rasterized(False)
    colorbar.solids.set_edgecolor("face")
    fig.text(0.227, 0.894, r"$-\log_{10}$ FDR", ha="right", va="center")

    local_peaks = pd.Series(peak_ids).str.extract(r"^(chr[^-]+)[-:](\d+)-(\d+)$").dropna()
    local_peaks.columns = ["chrom", "start", "end"]
    local_peaks[["start", "end"]] = local_peaks[["start", "end"]].astype(int)
    for peak in local_peaks.loc[local_peaks.chrom.eq(CHROM) & local_peaks.end.gt(WINDOW[0])
                                & local_peaks.start.lt(WINDOW[1])].itertuples(index=False):
        peaks.add_patch(Rectangle((peak.start, 0.15), peak.end - peak.start, 0.7, color="#ff6f69", lw=0))
    peaks.axvspan(*PEAK, color=DOMAIN_COLORS["D6"], alpha=0.22, lw=0)
    peaks.set_ylim(0, 1)
    peaks.text(-0.03, 0.5, "Peak", transform=peaks.transAxes, ha="right", va="center")
    draw_prdm1_logo(logo, pfm)
    for hit in hits.itertuples(index=False):
        midpoint = (hit.start + hit.end) / 2
        peaks.plot([midpoint, midpoint], [0.04, -0.27], color=DOMAIN_COLORS["D6"], lw=0.8, clip_on=False)
        for target in ((0.625, 0.835), (0.645, 0.745)):
            fig.add_artist(ConnectionPatch((midpoint, 0.5), target, "data", "figure fraction",
                                           axesA=peaks, lw=0.55, ls="--", color="#777777"))

    ymax = tracks.filter(like="_mean_fragment_cp10k").to_numpy().max() * 1.08
    for ax, domain in zip(coverage, DOMAINS):
        color = DOMAIN_COLORS[domain]
        values = tracks[f"{domain}_mean_fragment_cp10k"]
        ax.fill_between(tracks.position, values, color=color, alpha=0.80, lw=0)
        ax.plot(tracks.position, values, color=color, lw=0.55)
        ax.axvspan(*PEAK, color=DOMAIN_COLORS["D6"], alpha=0.07, lw=0)
        ax.set_ylim(0, ymax)
        ax.text(-0.03, 0.36, domain, transform=ax.transAxes, ha="right", va="center")

    for row in gene_model.loc[gene_model.feature.isin(["exon", "CDS"])].itertuples(index=False):
        gene.add_patch(Rectangle((row.start, 0.24), row.end - row.start,
                                 0.40 if row.feature == "exon" else 0.60, color="#222222", lw=0))
    far_end = gene_span.start if gene_span.strand == "-" else gene_span.end
    gene.annotate("", xy=(far_end, 0.45), xytext=(tss, 0.45),
                  arrowprops={"arrowstyle": "-|>", "color": "#222222", "lw": 0.8})
    gene.plot([tss, tss], [0.10, 0.84], color="#333333", lw=0.7)
    gene.text((gene_span.start + gene_span.end) / 2, 0.90, GENE, ha="center")
    gene.text(-0.03, 0.45, "Gene", transform=gene.transAxes, ha="right", va="center")
    gene.set_ylim(0, 1.25)
    gene.set_xlabel(f"{CHROM}:{WINDOW[0]:,}–{WINDOW[1]:,}", labelpad=5)
    for ax in [arc, peaks, *coverage, gene]:
        ax.set(xticks=[], yticks=[])
        ax.spines[:].set_visible(False)
        ax.grid(False)
    gene.spines["bottom"].set_visible(True)

    rng = np.random.RandomState(8)
    for index, domain in enumerate(DOMAINS):
        values = rna.loc[rna.domain.eq(domain), "Pou3f2_log1p_count"].to_numpy()
        y = len(DOMAINS) - index - 1
        color = DOMAIN_COLORS[domain]
        violin = expression.violinplot([values], positions=[y], widths=0.70, vert=False,
                                       showextrema=False, showmeans=False, showmedians=False)
        body = violin["bodies"][0]
        body.set(facecolor=color, edgecolor="#999999", linewidth=0.55, alpha=0.40)
        positive = values[values > 0]
        expression.scatter(positive, y + rng.normal(0, 0.045, len(positive)), s=1.1, color=color, lw=0)
        median = np.median(values)
        expression.plot([median, median], [y - 0.12, y + 0.12], color="black", lw=0.9)
    expression.set(xlim=(-0.08, 3.15), ylim=(-0.45, 4.65), yticks=[], xticks=[0, 2],
                   xlabel="Pou3f2 expression\nlog1p(raw RNA count)")
    expression.spines[["top", "right", "left"]].set_visible(False)
    expression.grid(False)
    fig.text(0.63, 0.038, f"Peak1: {CHROM}:{PEAK[0]:,}–{PEAK[1]:,}", ha="center")
    for coordinate, target in zip(PEAK, (0.54, 0.755)):
        fig.add_artist(ConnectionPatch((coordinate, 0), (target, 0.068), "data", "figure fraction",
                                       axesA=coverage[-1], lw=0.65, ls="--", color="#555555"))
    return fig
