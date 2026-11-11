"""Recompute P22 Fig. 4e, g and h from paired RNA counts and GO annotations."""

import gzip
import re
from dataclasses import dataclass
from io import StringIO

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.font_manager import FontProperties
from matplotlib.patches import ConnectionPatch, Patch, Rectangle
from matplotlib.textpath import TextToPath
from matplotlib.ticker import FixedLocator, FormatStrFormatter, FuncFormatter, MaxNLocator, MultipleLocator
from scipy import sparse
from scipy.ndimage import gaussian_filter
from scipy.stats import false_discovery_control, hypergeom, mannwhitneyu, rankdata, wilcoxon, zscore

from seareg_tutorial_regulatory import GROUPS, GROUPS_ATAC, GROUPS_H3, register_h3_to_atac


REGIONS = tuple(GROUPS)
COLORS = dict(zip(REGIONS, ("#7457C8", "#D84F9B", "#25A9B5")))
RNA_STYLE = {
    "font.family": ["Liberation Sans", "DejaVu Sans"],
    "font.size": 8, "axes.linewidth": 0.5, "axes.grid": False,
    "pdf.fonttype": 42, "svg.fonttype": "none",
}
BOOTSTRAP_RESAMPLES = 5000
BOOTSTRAP_SEED = 20260720


@dataclass
class RNASection:
    counts: sparse.csr_matrix
    genes: pd.Index
    barcodes: pd.Index
    domains: pd.Series
    xy: np.ndarray
    scale: np.ndarray

    def log_expression(self, row):
        return np.log1p(self.counts[row].toarray().ravel() * self.scale)


def load_section_rna(path, labels, xy):
    """Align a GEO gene-by-spot RNA count matrix to the adopted spot labels."""
    genes = []
    row_indices = []
    row_counts = []
    with gzip.open(path, "rt") as source:
        barcodes = pd.Index(source.readline().rstrip("\r\n").split("\t")).str.split("-").str[0]
        spot_ids = pd.Index(labels["spot_id"]).str.split("-").str[0]
        if not barcodes.is_unique or not spot_ids.is_unique or set(barcodes) != set(spot_ids):
            raise ValueError(f"RNA counts and adopted domain labels have different spot IDs: {path}")
        order = spot_ids.get_indexer(barcodes)
        domains = pd.Series(labels["domain"].to_numpy()[order], index=barcodes)
        library = np.zeros(len(barcodes), dtype=np.float64)
        indptr = [0]
        for line in source:
            gene, _, fields = line.partition("\t")
            counts = np.fromstring(fields, dtype=np.float32, sep="\t")
            if counts.size != len(barcodes):
                raise ValueError(f"{gene}: expected {len(barcodes)} RNA values, found {counts.size}")
            keep = np.flatnonzero(counts)
            genes.append(gene)
            row_indices.append(keep.astype(np.int32))
            row_counts.append(counts[keep])
            indptr.append(indptr[-1] + len(keep))
            library += counts
    gene_index = pd.Index(genes, name="gene")
    if not gene_index.is_unique or (library <= 0).any():
        raise ValueError(f"Duplicate gene names or zero-library spots in {path}")
    counts = sparse.csr_matrix(
        (np.concatenate(row_counts), np.concatenate(row_indices), indptr),
        shape=(len(genes), len(barcodes)),
    )
    return RNASection(counts, gene_index, barcodes, domains, np.asarray(xy)[order], 1e4 / library)


def _domain_profiles(section, prefix):
    # The original RNA-identity analysis stores per-gene statistics as float32.
    n_genes, n_spots = section.counts.shape
    domains = sorted(section.domains.unique(), key=lambda label: int(label[1:]))
    codes = pd.Index(domains).get_indexer(section.domains)
    domain_names = [prefix + label[1:] for label in domains]
    sizes = np.bincount(codes).astype(np.float64)
    means = np.empty((n_genes, len(sizes)), dtype=np.float32)
    effects = np.empty_like(means)
    variances = np.empty(n_genes, dtype=np.float32)
    for row in range(n_genes):
        values = section.log_expression(row)
        sums = np.bincount(codes, weights=values)
        means[row] = sums / sizes
        outside = (values.sum(dtype=np.float64) - sums) / (n_spots - sizes)
        effects[row] = sums / sizes - outside
        variances[row] = values.var()
    return {
        "means": pd.DataFrame(means, index=section.genes, columns=domain_names),
        "effects": pd.DataFrame(effects, index=section.genes, columns=domain_names),
        "sizes": pd.Series(sizes, index=domain_names),
        "detection": pd.Series(np.diff(section.counts.indptr) / n_spots, index=section.genes),
        "variance": pd.Series(variances, index=section.genes),
    }


def _primary_genes(atac, h3):
    shared = atac["detection"].index.intersection(h3["detection"].index)
    technical = shared.str.match(r"^(mt[.-]|Rpl\d|Rps\d)", case=False)
    expressed = shared[(atac["detection"].loc[shared] >= .01)
                       & (h3["detection"].loc[shared] >= .01) & ~technical]
    first = atac["variance"].loc[expressed].nlargest(1500).index
    second = h3["variance"].loc[expressed].nlargest(1500).index
    return first.union(second).sort_values()


def _regional_effect(profile, members):
    means = profile["means"]
    sizes = profile["sizes"]
    outside = means.columns.difference(members, sort=False)
    inside_mean = means[members].mul(sizes[members], axis=1).sum(axis=1) / sizes[members].sum()
    outside_mean = means[outside].mul(sizes[outside], axis=1).sum(axis=1) / sizes[outside].sum()
    return inside_mean - outside_mean


def _program_scores(section, genes):
    rows = section.genes.get_indexer(genes)
    expression = np.vstack([section.log_expression(row) for row in rows]).astype(np.float32)
    scale = expression.std(axis=1, keepdims=True)
    scale[scale < 1e-8] = 1
    return ((expression - expression.mean(axis=1, keepdims=True)) / scale).mean(axis=0)


def compute_rna_identity(atac_rna, h3_rna):
    """Compare 13 × 11 domain profiles and recompute six regional RNA maps."""
    atac = _domain_profiles(atac_rna, "A")
    h3 = _domain_profiles(h3_rna, "H")
    genes = _primary_genes(atac, h3)
    a = atac["effects"].loc[genes].to_numpy(dtype=np.float64)
    h = h3["effects"].loc[genes].to_numpy(dtype=np.float64)
    a_rank = zscore(rankdata(a, axis=0), axis=0, ddof=1)
    h_rank = zscore(rankdata(h, axis=0), axis=0, ddof=1)
    matrix = pd.DataFrame(
        a_rank.T @ h_rank / (len(genes) - 1),
        index=atac["effects"].columns.rename("ATAC_domain"),
        columns=h3["effects"].columns.rename("H3K27ac_domain"),
    )

    aligned_h3 = register_h3_to_atac(atac_rna.xy, h3_rna.xy)
    bounds = np.vstack([atac_rna.xy, aligned_h3])
    lower, upper = bounds.min(axis=0), bounds.max(axis=0)
    xr, yr = zip(lower, upper)

    programs = {}
    grids = {}
    all_scores = []
    for region, (atac_domains, h3_domains) in GROUPS.items():
        a_effect = _regional_effect(atac, ["A" + label[1:] for label in atac_domains]).loc[genes]
        h_effect = _regional_effect(h3, ["H" + label[1:] for label in h3_domains]).loc[genes]
        a_top = a_effect[a_effect > 0].sort_values(ascending=False).head(100).index
        h_top = h_effect[h_effect > 0].sort_values(ascending=False).head(100).index
        program = a_top.intersection(h_top).sort_values()
        programs[region] = program.tolist()
        for label, section, coords in [("ATAC-RNA", atac_rna, atac_rna.xy),
                                       ("H3K27ac-RNA", h3_rna, aligned_h3)]:
            scores = _program_scores(section, program)
            all_scores.append(scores)
            hist, y_edges, x_edges = np.histogram2d(coords[:, 1], coords[:, 0], bins=72, range=[yr, xr])
            weighted, _, _ = np.histogram2d(
                coords[:, 1], coords[:, 0], bins=[y_edges, x_edges], weights=scores,
            )
            smooth_count = gaussian_filter(hist, 1.15)
            grid = np.divide(
                gaussian_filter(weighted, 1.15), smooth_count,
                out=np.full_like(smooth_count, np.nan), where=smooth_count > .035,
            )
            grids[region, label] = {"values": grid, "x_edges": x_edges, "y_edges": y_edges}
    return {
        "matrix": matrix,
        "grids": grids,
        "programs": programs,
        "primary_genes": genes,
        "map_limits": np.quantile(np.concatenate(all_scores), [.02, .98]),
        "domain_profiles": (atac, h3),
    }


def _readback_profile(frame):
    # The publication statistics consumed decimal TSV profiles. Preserve their
    # precision in memory because rounding can resolve a Wilcoxon rank tie.
    return pd.read_csv(StringIO(frame.to_csv(sep="\t")), sep="\t", index_col=0, float_precision="high")


def target_rna_effects(identity):
    """Calculate the two-section target RNA gates from freshly computed profiles."""
    atac, h3 = [
        {"means": _readback_profile(profile["means"]), "sizes": profile["sizes"]}
        for profile in identity["domain_profiles"]
    ]
    common = atac["means"].index.intersection(h3["means"].index)
    rows = []
    for region, (a_domains, h_domains) in GROUPS.items():
        a_effect = _regional_effect(atac, ["A" + label[1:] for label in a_domains])
        h_effect = _regional_effect(h3, ["H" + label[1:] for label in h_domains])
        rows.append(pd.DataFrame({
            "atac_group": region, "target_gene": common,
            "target_ATAC_RNA_effect": a_effect.loc[common].to_numpy(),
            "target_H3_RNA_effect": h_effect.loc[common].to_numpy(),
        }))
    return pd.concat(rows, ignore_index=True)


def compute_tf_rna_tests(atac_rna, motif_to_tf):
    """Test regional TF expression on raw ATAC-section paired RNA counts."""
    gene_rows = {gene.lower(): row for row, gene in enumerate(atac_rna.genes)}
    masks = {region: atac_rna.domains.isin(domains).to_numpy() for region, domains in GROUPS_ATAC.items()}
    rows = []
    for gene in sorted(motif_to_tf.candidate_tf_gene.dropna().unique()):
        row = gene_rows.get(gene.lower())
        if row is None:
            continue
        counts = atac_rna.counts[row].toarray().ravel()
        for region, inside in masks.items():
            target, outside = counts[inside], counts[~inside]
            test = mannwhitneyu(target, outside, alternative="greater")
            rows.append((region, gene, target.mean() - outside.mean(), test.pvalue, len(target), len(outside)))
    tests = pd.DataFrame(rows, columns=["group", "gene", "rna_delta_mean", "rna_p", "n_in", "n_out"])
    tests.insert(4, "rna_fdr", tests.groupby("group").rna_p.transform(false_discovery_control))
    return _readback_profile(tests)


def _target_region_z(section, genes, groups):
    # Resolve case-insensitive gene symbols by taking the first count-matrix row.
    first = {}
    for row, gene in enumerate(section.genes.str.upper()):
        first.setdefault(gene, row)
    masks = {region: section.domains.isin(members).to_numpy() for region, members in groups.items()}
    profiles = pd.DataFrame(index=genes, columns=REGIONS, dtype=float)
    for gene in genes:
        if gene not in first:
            continue
        expression = section.log_expression(first[gene])
        mean, sd = expression.mean(), expression.std(ddof=0)
        if mean > 0 and sd > 0:
            profiles.loc[gene] = [(expression[mask].mean() - mean) / sd for mask in masks.values()]
    return profiles


def compute_target_rna(atac_rna, h3_rna, early_targets):
    """Compute pooled two-section RNA z scores, bootstrap bars and paired tests."""
    genes = pd.Index(early_targets.target_upper.unique())
    atac = _readback_profile(_target_region_z(atac_rna, genes, GROUPS_ATAC))
    h3 = _readback_profile(_target_region_z(h3_rna, genes, GROUPS_H3))
    pooled = ((atac + h3) / 2).dropna().rename_axis("gene")
    members = early_targets.rename(columns={"target_upper": "gene"}).drop_duplicates()
    values = members.join(pooled, on="gene", how="inner").sort_values(["target_set", "gene"])
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    bars, tests = [], []
    for group in REGIONS:
        region_values = values.loc[values.target_set.eq(group)]
        for region in REGIONS:
            sample = region_values[region].to_numpy()
            sampled = sample[rng.integers(0, len(sample), size=(BOOTSTRAP_RESAMPLES, len(sample)))]
            lo, hi = np.quantile(np.median(sampled, axis=1), [.025, .975])
            bars.append((group, region, len(sample), np.median(sample), lo, hi, BOOTSTRAP_RESAMPLES, BOOTSTRAP_SEED))
            if region != group:
                corresponding = region_values[group].to_numpy()
                test = wilcoxon(corresponding, sample, alternative="greater", zero_method="wilcox")
                tests.append((group, region, len(sample), np.median(corresponding - sample), test.pvalue))
    bars = pd.DataFrame(bars, columns=[
        "target_set", "expression_region", "n_genes", "median", "CI95_low", "CI95_high",
        "bootstrap_resamples", "seed",
    ])
    tests = pd.DataFrame(tests, columns=["target_set", "comparator_region", "n_genes", "effect", "p_value"])
    tests["BH_q"] = false_discovery_control(tests.p_value)
    long = values.melt(
        id_vars=["target_set", "gene"], value_vars=REGIONS, var_name="expression_region",
        value_name="mean_two_sections_RNA_z",
    )
    return long, bars, tests


def compute_target_go(early_targets, p2g_symbols, go_gmt):
    """Test enrichment against annotated genes from the official P2G universe."""
    with open(go_gmt, encoding="utf-8") as source:
        annotation = {}
        for line in source:
            term, _, *genes = line.rstrip("\n").split("\t")
            annotation[term] = {gene.upper() for gene in genes if gene}
    background = {str(gene).upper() for gene in p2g_symbols if pd.notna(gene)}
    background &= set().union(*annotation.values())
    terms = {term: annotated & background for term, annotated in annotation.items()}
    rows = []
    for region in REGIONS:
        targets = set(early_targets.loc[early_targets.target_set.eq(region), "target_upper"]) & background
        for term, annotated in terms.items():
            if 10 <= len(annotated) <= 500:
                rows.append((region, term, len(targets), len(targets & annotated), len(annotated)))
    tests = pd.DataFrame(rows, columns=["region", "GO_term", "target_n", "target_hits", "comparator_hits"])
    tests["background_n"] = len(background)
    tests["p_value"] = hypergeom.sf(
        tests.target_hits - 1, tests.background_n, tests.comparator_hits, tests.target_n,
    )
    fold = (tests.target_hits / tests.target_n) / (tests.comparator_hits / tests.background_n)
    tests["effect"] = np.log2(fold.where(fold.gt(0)))
    tests["q_value"] = false_discovery_control(tests.p_value)
    tests["region_q"] = tests.groupby("region", sort=False).p_value.transform(false_discovery_control)

    summary = tests.loc[tests.q_value.lt(.05)].copy()
    summary["GO_ID"] = summary.GO_term.str.extract(r"(GO:\d+)", expand=False)
    summary = summary.sort_values(["region", "q_value", "p_value", "GO_ID"])
    summary["rank_within_gene_set"] = summary.groupby("region", sort=False).cumcount() + 1
    displayed = summary.loc[summary.target_hits.ge(3)].sort_values(["region", "q_value", "p_value", "GO_term"])
    displayed = displayed.groupby("region", sort=False).head(3)
    summary["display_order"] = displayed.groupby("region", sort=False).cumcount() + 1
    summary["displayed_in_main"] = summary.display_order.notna()
    summary = summary.loc[summary.rank_within_gene_set.le(50) | summary.displayed_in_main].rename(columns={
        "effect": "log2_fold_enrichment", "q_value": "BH_q_global_6351",
        "target_n": "target_n_mapped", "comparator_hits": "background_term_hits",
    })
    return tests, summary


def plot_rna_identity(result):
    """Draw the Fig. 4e heatmap and six paired RNA program maps."""
    import seaborn as sns

    matrix = result["matrix"]
    grids = result["grids"]
    with mpl.rc_context(RNA_STYLE):
        fig = plt.figure(figsize=(5.2, 2.96))
        ax = fig.add_axes([.11, .18, .445, .75])
        scc_bar = fig.add_axes([.568, .18, .017, .19])
        sns.heatmap(
            matrix, ax=ax, cbar_ax=scc_bar, cmap="vlag", center=0,
            vmin=-.6, vmax=.9, linewidths=.15, linecolor="white",
            cbar_kws={"ticks": [-.5, 0, .5]},
        )
        ax.set(xlabel="H3K27ac domains", ylabel="ATAC domains")
        ax.tick_params(length=0, pad=1, labelsize=7)
        plt.setp(ax.get_xticklabels(), rotation=0)
        plt.setp(ax.get_yticklabels(), rotation=0)
        ax.grid(False)
        scc_bar.set_title("RNA\nSCC", fontsize=8, loc="left", pad=2)
        scc_bar.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
        scc_bar.tick_params(length=2, pad=1, labelsize=7)
        for collection in scc_bar.collections:
            collection.set_rasterized(False)
        scc_bar.spines["outline"].set_visible(True)

        lower, upper = result["map_limits"]
        for row, region in enumerate(REGIONS):
            color = COLORS[region]
            a_domains, h_domains = GROUPS[region]
            x = matrix.columns.get_loc("H" + h_domains[0][1:])
            y = matrix.index.get_loc("A" + a_domains[0][1:])
            ax.add_patch(Rectangle((x, y), len(h_domains), len(a_domains), fill=False, ec=color, lw=1))
            for col, (source, title) in enumerate([("ATAC-RNA", "ATAC"), ("H3K27ac-RNA", "H3K27ac")]):
                panel = fig.add_axes([.705 + .151 * col, .682 - .23 * row, .126, .215])
                grid = grids[region, source]
                mesh = panel.pcolormesh(
                    grid["x_edges"], grid["y_edges"], grid["values"], cmap="magma", shading="flat",
                    vmin=lower, vmax=upper, edgecolors="face", antialiased=False,
                )
                panel.set_aspect("equal")
                panel.set_xticks([])
                panel.set_yticks([])
                for spine in panel.spines.values():
                    spine.set_color(color)
                if row == 0:
                    panel.set_title(title, pad=2)
                if col == 0:
                    panel.set_ylabel(region, labelpad=1)
                    link = ConnectionPatch(
                        xyA=(x + len(h_domains), y + len(a_domains) / 2), coordsA="data", axesA=ax,
                        xyB=(-.25, .5), coordsB="axes fraction", axesB=panel, clip_on=False,
                        color="#161616", linewidth=.55, linestyle=(0, (5, 5)),
                    )
                    fig.add_artist(link)

        rna_bar = fig.add_axes([.795, .107, .105, .032])
        bar = fig.colorbar(mesh, cax=rna_bar, orientation="horizontal", ticks=[-.4, .6])
        bar.solids.set_rasterized(False)
        rna_bar.tick_params(length=2, pad=1, labelsize=7)
        rna_bar.set_title("Mean RNA z-score", fontsize=8, pad=2)
        return fig


def plot_target_rna(bars):
    """Draw the adopted Fig. 4g grouped bars with gene-bootstrap intervals."""
    colors = dict(zip(REGIONS, ("#75619C", "#C75794", "#369EAB")))
    with mpl.rc_context(RNA_STYLE):
        fig = plt.figure(figsize=(152 / 72, 113 / 72))
        ax = fig.add_axes((31.5 / 152, 14 / 113, 118 / 152, 82 / 113))
        data = bars.set_index(["target_set", "expression_region"])
        for i, group in enumerate(REGIONS):
            for j, region in enumerate(REGIONS):
                row = data.loc[group, region]
                x = i + (j - 1) * .22
                median = row["median"]
                ax.bar(x, median, width=.19, color=colors[region], alpha=.70, edgecolor="#252525", linewidth=.55)
                ax.errorbar(
                    x, median, yerr=[[median - row.CI95_low], [row.CI95_high - median]],
                    fmt="none", ecolor="#282828", lw=.55, capsize=1.2, capthick=.55,
                )
        lower = min(-.09, float(bars.CI95_low.min()) - .018)
        upper = max(.54, float(bars.CI95_high.max()) * 1.10)
        ax.set(ylim=(lower, upper), xlim=(-.54, 2.42), xticks=range(3), xticklabels=REGIONS, ylabel="RNA z-score")
        ticks = FixedLocator([0, .2, .4]) if (lower, upper) == (-.09, .54) else MaxNLocator(nbins=4)
        ax.yaxis.set_major_locator(ticks)
        ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
        ax.spines[["right", "top"]].set_visible(False)
        ax.tick_params(length=2.2, width=.5, pad=2)
        ax.yaxis.labelpad = 1
        ax.legend(
            handles=[Patch(facecolor=color, edgecolor=color, alpha=.70, label=group) for group, color in colors.items()],
            loc="upper left", borderpad=.05, borderaxespad=.65, labelspacing=0,
            handlelength=.9, handleheight=.5, handletextpad=.4, frameon=False,
        )
        fig.text(.60, 1 - 8.1 / 113, "Regional target genes", ha="center", va="center")
        return fig


def plot_target_go(summary):
    """Draw Fig. 4h using only terms eligible for the main-panel display."""
    with mpl.rc_context(RNA_STYLE):
        fig = plt.figure(figsize=(181 / 72, 122 / 72))
        ax = fig.add_axes((37.5 / 181, 23 / 122, 127.5 / 181, 86 / 122))
        shown = summary.loc[summary.displayed_in_main]
        max_fold = max([0.0, *shown.log2_fold_enrichment])
        ax.set(xlim=(0, max(5, max_fold * 1.07)), ylim=(-.75, 9.75), yticks=[])
        ax.xaxis.set_major_locator(MultipleLocator(2))
        ax.tick_params(axis="x", length=2.4, width=.5, pad=1.5)
        ax.spines[["left", "right"]].set_visible(False)
        text_path = TextToPath()
        label_font = FontProperties(family="Liberation Sans", size=6)
        for i, group in enumerate(REGIONS):
            top = 9 - 3.45 * i
            region = shown.loc[shown.region.eq(group)].sort_values("display_order")
            center = top - (len(region) - 1) / 2 if len(region) else top - 1
            ax.text(-.03, center, group, ha="right", va="center", transform=ax.get_yaxis_transform())
            for j, row in enumerate(region.itertuples(index=False)):
                y = top - j
                ax.barh(y, row.log2_fold_enrichment, height=.67, color=COLORS[group], alpha=.52, linewidth=0)
                label = re.sub(r"\s+\(GO:\d+\)$", "", row.GO_term).capitalize().replace("-coa", "-CoA")
                width, _, _ = text_path.get_text_width_height_descent(label, label_font, ismath=False)
                ax.text(.035, y, label, va="center", fontsize=min(6, 840 / max(width, 1)), clip_on=False)
        fig.text(.50, 1 - 8.1 / 122, "GO enrichment of target genes", ha="center", va="center")
        fig.text(.51, 5.4 / 122, "Log$_2$ fold enrichment", ha="center", va="center")
        return fig
