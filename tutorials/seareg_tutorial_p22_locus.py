"""Recompute the P22 SOX6–Mal locus browser in Fig. 4j."""

import gzip
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d

from .seareg_tutorial_regulatory import (
    GROUPS_ATAC,
    GROUPS_H3,
    canonical_peak,
    read_table,
    register_h3_to_atac,
    spot_keys,
)

CHROM = "chr2"
WINDOW = (127596695, 127671695)
ATAC_PEAK = (127627255, 127628224)
H3_PEAK = (127626955, 127627859)
MOTIF = "Sox6_MA0515_1"
REGION_COLORS = dict(zip(GROUPS_ATAC, ("#7457C8", "#D84F9B", "#25A9B5")))


def _region_lookup(labels, groups):
    domains = {domain: group for group, members in groups.items() for domain in members}
    return labels.set_index("spot_id").domain.map(domains).dropna()


def _fragment_lines(path):
    """Retrieve the locus from a bgzip fragment table and its adjacent index."""
    left, right = WINDOW
    if shutil.which("tabix"):
        interval = f"{CHROM}:{left + 1}-{right}"
        yield from subprocess.run(
            ["tabix", str(path), interval], check=True, capture_output=True, text=True,
        ).stdout.splitlines()
    else:
        try:
            import pysam
        except ImportError as exc:
            raise RuntimeError("Fig. 4j needs tabix or pysam to read indexed fragments.") from exc
        with pysam.TabixFile(str(path)) as indexed:
            yield from indexed.fetch(CHROM, left, right)


def _coverage(assay, fragment_path, library_path, regions):
    left, right = WINDOW
    bins = 500
    edges = np.linspace(left, right, bins + 1)
    centers = (edges[:-1] + edges[1:]) / 2
    library = read_table(library_path).set_index("barcode").total_fragments
    library.index = spot_keys(library.index)
    denominators = library.reindex(regions.index)
    if denominators.isna().any() or not np.isfinite(denominators).all() or denominators.le(0).any():
        raise ValueError(f"Missing positive {assay} fragment library size for a regional spot.")
    coverage_delta = {group: np.zeros(bins + 1) for group in REGION_COLORS}
    for line in _fragment_lines(fragment_path):
        _, start, end, barcode, multiplicity, *_ = line.split("\t")
        spot = barcode.split("-", 1)[0]
        group = regions.get(spot)
        if group is None:
            continue
        lo, hi = max(int(start), left), min(int(end), right)
        if lo >= hi:
            continue
        first = (lo - left) * bins // (right - left)
        last = ((hi - left) * bins + (right - left) - 1) // (right - left)
        weight = float(multiplicity) * 10000 / denominators.at[spot]
        coverage_delta[group][first] += weight
        coverage_delta[group][last] -= weight

    sizes = regions.value_counts()
    mean_coverage = {
        group: gaussian_filter1d(np.cumsum(delta[:-1]) / sizes.at[group], sigma=1.15)
        for group, delta in coverage_delta.items()
    }
    shared_max = max(profile.max() for profile in mean_coverage.values())
    if not np.isfinite(shared_max) or shared_max <= 0:
        raise ValueError(f"No positive {assay} locus coverage after joining fragments to spots.")
    return pd.concat([
        pd.DataFrame({
            "assay": assay, "group": group, "chrom": CHROM, "bin_center": centers,
            "mean_fragment_coverage_per_spot_cp10k": profile,
            "shared_assay_scaled": profile / shared_max, "assay_shared_max_cp10k": shared_max,
            "n_group_spots": int(sizes.at[group]),
        })
        for group, profile in mean_coverage.items()
    ], ignore_index=True)


def _mal_rna(path, spot_ids):
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as stream:
        barcodes = stream.readline().rstrip("\n").split("\t")
        barcodes = pd.Index([barcode.split("-", 1)[0] for barcode in barcodes if barcode])
        for line in stream:
            gene, counts = line.split("\t", 1)
            if gene.casefold() == "mal":
                values = np.fromstring(counts, sep="\t", dtype=np.float32)
                break
        else:
            raise ValueError(f"No Mal row in {path}.")
    result = pd.Series(np.log1p(values), index=barcodes).reindex(spot_ids)
    if result.isna().any():
        raise ValueError(f"Missing Mal RNA values for {int(result.isna().sum())} model spots in {path}.")
    return result


def _embedding_score(embeddings, kind, feature):
    """Use the original float32 spot–feature cosine followed by whole-section z-scoring."""
    spots = embeddings["spot"].to_numpy(dtype=np.float32)
    features = embeddings[kind]
    match = features.index.map(canonical_peak) == canonical_peak(feature)
    if match.sum() != 1:
        raise ValueError(f"Expected one {kind} embedding for {feature}.")
    vector = features.loc[match].to_numpy(dtype=np.float32)[0]
    spots = spots / np.maximum(np.linalg.norm(spots, axis=1, keepdims=True), 1e-12)
    cosine = spots @ (vector / max(float(np.linalg.norm(vector)), 1e-12))
    return (cosine - cosine.mean()) / max(float(cosine.std()), 1e-12)


def compute_locus(atac, atac_labels, atac_xy, h3, h3_labels, h3_xy, evidence):
    """Build the observed tracks, paired RNA, feature maps, P2G arcs and motif logo."""
    atac_regions = _region_lookup(atac_labels, GROUPS_ATAC)
    h3_regions = _region_lookup(h3_labels, GROUPS_H3)
    tracks = pd.concat([
        _coverage("ATAC", evidence["locus_atac_fragments"], evidence["locus_atac_library"], atac_regions),
        _coverage("H3K27ac", evidence["locus_h3_fragments"], evidence["locus_h3_library"], h3_regions),
    ], ignore_index=True)
    atac_rna = _mal_rna(evidence["atac_rna"], atac_labels.spot_id)
    h3_rna = _mal_rna(evidence["h3_rna"], h3_labels.spot_id)
    rna_points = pd.concat([
        pd.DataFrame({"section": name, "group": groups, "value": values.reindex(groups.index)})
        for name, groups, values in (("ATAC–RNA", atac_regions, atac_rna), ("H3K27ac–RNA", h3_regions, h3_rna))
    ]).rename_axis("spot_id").reset_index()

    h3_aligned = register_h3_to_atac(atac_xy, h3_xy)
    features = (
        ("SOX6 motif MSS", atac_labels, atac_xy, _embedding_score(atac, "motif", MOTIF)),
        ("ATAC peak PSS", atac_labels, atac_xy, _embedding_score(atac, "peak", f"{CHROM}-{ATAC_PEAK[0]}-{ATAC_PEAK[1]}")),
        ("H3K27ac peak PSS", h3_labels, h3_aligned, _embedding_score(h3, "peak", f"{CHROM}-{H3_PEAK[0]}-{H3_PEAK[1]}")),
        ("Mal RNA", atac_labels, atac_xy, atac_rna.to_numpy()),
    )
    maps = pd.concat([
        labels.assign(feature=name, x=xy[:, 0], y=xy[:, 1], value=values)
        for name, labels, xy, values in features
    ], ignore_index=True)
    links = read_table(evidence["p2g"])
    in_window = ((links.peak_start + links.peak_end) / 2).between(*WINDOW)
    links = links.loc[links.symbol.str.casefold().eq("mal") & links.peak_chr.eq(CHROM) & in_window].sort_values("FDR")
    links = links.assign(selected=links.peak_start.eq(ATAC_PEAK[0]) & links.peak_end.eq(ATAC_PEAK[1]))
    if links.selected.sum() != 1:
        raise ValueError("Official P2G must contain the selected SOX6-associated Mal peak.")
    model = read_table(evidence["locus_gene_model"])
    logo = read_table(evidence["locus_pfm"], usecols=["position", "base", "PFM_count"])
    logo = logo.sort_values(["position", "base"])
    probability = (logo.PFM_count + .5) / (logo.groupby("position").PFM_count.transform("sum") + 2.)
    information = 2 + (probability * np.log2(probability)).groupby(logo.position).transform("sum")
    logo = logo.assign(display_letter_height_bits=probability * information)
    return {"tracks": tracks, "rna_points": rna_points, "maps": maps,
            "links": links, "logo": logo, "gene_model": model}


def _logo(axis, table):
    from matplotlib.font_manager import FontProperties
    from matplotlib.patches import PathPatch
    from matplotlib.textpath import TextPath
    from matplotlib.transforms import Affine2D

    palette = {"A": "#2CA02C", "C": "#1F77B4", "G": "#F39C12", "T": "#D62728"}
    font = FontProperties(family="DejaVu Sans", weight="bold")
    for position, bases in table.groupby("position"):
        bottom = 0.
        for row in bases.sort_values("display_letter_height_bits").itertuples(index=False):
            height = row.display_letter_height_bits
            if height <= 1e-5:
                continue
            glyph = TextPath((0, 0), row.base, size=1, prop=font)
            box = glyph.get_extents()
            sx, sy = .86 / box.width, height / box.height
            offset = Affine2D().scale(sx, sy).translate(position - .43 - box.x0 * sx, bottom - box.y0 * sy)
            axis.add_patch(PathPatch(glyph, transform=offset + axis.transData, facecolor=palette[row.base], edgecolor="none"))
            bottom += height
    axis.set(xlim=(.45, 10.55), ylim=(0, 2), xticks=range(1, 11), yticks=[0, 1, 2], ylabel="Bits")
    axis.tick_params(labelsize=5.5, length=1.5, pad=1)
    axis.spines[["top", "right"]].set_visible(False)
    axis.set_title("SOX6 motif", fontsize=8, pad=2)


def _draw_coverage(axis, track_table, assay, tss):
    peak = ATAC_PEAK if assay == "ATAC" else H3_PEAK
    axis.axvspan(*peak, color="#E5904B", alpha=.14)
    axis.axvline(tss, color="#666666", ls=":", lw=.55)
    for level, (group, color) in zip((2., 1., 0.), REGION_COLORS.items()):
        local = track_table.loc[track_table.group.eq(group)].sort_values("bin_center")
        axis.fill_between(local.bin_center, level, level + .83 * local.shared_assay_scaled, color=color, lw=0)
        axis.text(-.016, level + .24, group, transform=axis.get_yaxis_transform(), ha="right", va="center", fontsize=8)
    axis.set(xlim=WINDOW, ylim=(-.10, 2.9), xticks=[], yticks=[])
    axis.spines[:].set_visible(False)
    axis.text(-.19, .53, assay, rotation=90, transform=axis.transAxes, ha="center", va="center", fontsize=9)


def _draw_rna(axis, points, section, rng):
    xmax = float(np.ceil(points.value.max()))
    for level, (group, color) in zip((2., 1., 0.), REGION_COLORS.items()):
        values = points.loc[points.group.eq(group), "value"].to_numpy()
        parts = axis.violinplot(values, positions=[level + .24], vert=False, widths=.60, showextrema=False, points=100)
        parts["bodies"][0].set(facecolor=color, edgecolor=color, alpha=.52, linewidth=.35)
        axis.scatter(values, level + .24 + rng.uniform(-.065, .065, len(values)),
                     s=.45, color=color, alpha=.22, linewidth=0, rasterized=False)
        median = np.median(values)
        axis.plot([median, median], [level + .11, level + .37], color="black", lw=.95)
    axis.set(xlim=(0, xmax * 1.04), ylim=(-.10, 2.9), yticks=[], xticks=[0, xmax / 2, xmax])
    axis.set_xticklabels([f"{x:g}" for x in [0, xmax / 2, xmax]])
    axis.spines[["left", "right", "top"]].set_visible(False)
    axis.tick_params(axis="x", labelsize=7, length=2, pad=1)
    axis.set_title(section, fontsize=7, pad=1)


def plot_locus(locus):
    """Draw the six measured coverage tracks, six RNA violins and four spatial maps."""
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from matplotlib.patches import PathPatch, Rectangle
    from matplotlib.path import Path as DrawPath

    fig = plt.figure(figsize=(7.2, 6.3))
    selected = locus["links"].loc[locus["links"].selected].iloc[0]
    tss = selected.gene_end if selected.strand == "-" else selected.gene_start
    fig.text(.52, .968, f"Selected Peak2Gene link   r = {selected.Correlation:.2f}, FDR = {selected.FDR:.2e}",
             ha="center", fontsize=10)
    browser = fig.add_axes([.19, .765, .60, .17])
    browser.set(xlim=WINDOW, ylim=(0, 1), xticks=[], yticks=[])
    browser.spines[:].set_visible(False)
    scale = mpl.colors.Normalize(0, 3, clip=True)
    reds = mpl.colormaps["Reds"]
    for row in locus["links"].itertuples(index=False):
        center = (row.peak_start + row.peak_end) / 2
        height = .12 + .50 * abs(center - tss) / (WINDOW[1] - WINDOW[0])
        curve = DrawPath([(center, .28), (center + (tss-center)*.45, .28 + height), (tss, .28)],
                         [DrawPath.MOVETO, DrawPath.CURVE3, DrawPath.CURVE3])
        color = reds(scale(-np.log10(max(row.FDR, 1e-300))))
        browser.add_patch(PathPatch(curve, facecolor="none", edgecolor=color, lw=1.1))
        browser.add_patch(Rectangle((row.peak_start, .18), row.peak_end-row.peak_start, .06,
                                    facecolor="#3F7FBF" if row.selected else color,
                                    edgecolor="#E5904B" if row.selected else "none", lw=.8))
    browser.add_patch(Rectangle((H3_PEAK[0], .055), H3_PEAK[1]-H3_PEAK[0], .06, color="#E5904B"))
    browser.vlines(np.mean(ATAC_PEAK), .24, .39, color="#7457C8", lw=.8)
    for label, level in (("ATAC peaks", .21), ("H3K27ac peaks", .085)):
        browser.text(-.016, level, label, transform=browser.get_yaxis_transform(), ha="right", va="center", fontsize=8)
    cax = browser.inset_axes([.04, .78, .20, .055])
    colorbar = fig.colorbar(mpl.cm.ScalarMappable(norm=scale, cmap=reds), cax=cax, orientation="horizontal", ticks=[0, 3])
    colorbar.ax.tick_params(labelsize=6, length=1.5, pad=1)
    colorbar.set_label(r"$-\log_{10}$ FDR", fontsize=7, labelpad=1)
    colorbar.solids.set_rasterized(False)
    logo_axis = fig.add_axes([.70, .821, .19, .102])
    _logo(logo_axis, locus["logo"])

    rng = np.random.default_rng(20260719)
    for assay, section, y, peak in (("ATAC", "ATAC–RNA", .579, ATAC_PEAK), ("H3K27ac", "H3K27ac–RNA", .370, H3_PEAK)):
        left = fig.add_axes([.19, y, .60, .167])
        _draw_coverage(left, locus["tracks"].loc[locus["tracks"].assay.eq(assay)], assay, tss)
        left.text(.015, -.095, f"{assay} peak   {CHROM}:{peak[0]:,}–{peak[1]:,}",
                  transform=left.transAxes, va="top", fontsize=8)
        right = fig.add_axes([.814, y, .155, .167])
        _draw_rna(right, locus["rna_points"].loc[locus["rna_points"].section.eq(section)], section, rng)
    right.set_xlabel("Mal expression (log1p count)", fontsize=7, labelpad=2)

    model = locus["gene_model"]
    axis = fig.add_axes([.19, .283, .60, .055])
    axis.set(xlim=WINDOW, ylim=(0, 1), xticks=[], yticks=[])
    axis.spines[["left", "right", "top"]].set_visible(False)
    start, end = model.start.min(), model.end.max()
    axis.hlines(.46, start, end, color="#222222", lw=.65)
    for x in np.linspace(start, end, 9)[1:-1]:
        axis.plot(x, .46, marker="<" if selected.strand == "-" else ">", ms=2.9, color="#222222", mew=0)
    for row in model.itertuples(index=False):
        height = .27 if row.feature == "CDS" else .13
        axis.add_patch(Rectangle((row.start, .46 - height/2), row.end-row.start+1, height,
                                 facecolor="#222222" if row.feature == "CDS" else "#909090",
                                 edgecolor="#222222", lw=.35))
    axis.text((start+end)/2, .74, "Mal", ha="center", fontsize=8)
    axis.text(-.016, .46, "gene", transform=axis.get_yaxis_transform(), ha="right", va="center", fontsize=8)
    axis.set_xlabel(f"{CHROM}:{WINDOW[0]:,}–{WINDOW[1]:,}", fontsize=8, labelpad=2)

    for i, (name, local) in enumerate(locus["maps"].groupby("feature", sort=False)):
        axis = fig.add_axes([.027 + i*.247, .015, .216, .222])
        artist = axis.scatter(local.x, local.y, c=local.value, cmap="viridis" if name == "Mal RNA" else "magma",
                              marker="s", linewidths=0, rasterized=False)
        axis.set_aspect("equal")
        axis.axis("off")
        axis.set_title(name, fontsize=8, pad=3)
        x_pitch = np.median(np.diff(np.unique(local.x)))
        y_pitch = np.median(np.diff(np.unique(local.y)))
        x0, x1 = axis.get_xlim()
        y0, y1 = axis.get_ylim()
        size = min(fig.get_figwidth() * .216 * x_pitch / (x1 - x0),
                   fig.get_figheight() * .222 * y_pitch / (y1 - y0)) * 72 * 1.05
        artist.set_sizes([size ** 2])
    for axis in fig.axes:
        axis.grid(False)
    return fig
