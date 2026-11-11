"""Draw Fig. 4d, f and i from the computed P22 tutorial results."""

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch, Rectangle
import networkx as nx
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter
from seaborn import color_palette

from seareg_tutorial_regulatory import GROUPS, register_h3_to_atac


MATRIX_OUTLINE_COLORS = {"CC/oligo": "#7457C8", "LV/VZ": "#D84F9B", "Cortex": "#4EB9C3"}
DENSITY_MAP_FRAME_COLORS = {"CC/oligo": "#6756A3", "LV/VZ": "#CA4C92", "Cortex": "#23A1AC"}
REGIONS = list(GROUPS)
BLOCKS = [(2, 2, 1, 1), (7, 7, 1, 1), (8, 8, 2, 4)]
TF_COLORS = dict(zip(
    ["Sox2", "Sox4", "Sox6", "Sox9", "Nfib", "Zfp281", "Nfix", "Sp1", "Klf7"],
    mpl.colormaps["tab10"].colors,
))
TARGET_LABELS = ("Mal", "Qk", "Cldn11", "Sox10", "Olig1", "Olig2", "Itpr2", "Enpp2", "Ndrg1", "Sox2ot")


def regional_density_grids(atac_labels, atac_xy, h3_labels, h3_xy):
    """Use the manuscript's 70-bin density maps beside the 80-bin matrix."""
    h3_aligned = register_h3_to_atac(atac_xy, h3_xy)
    grids = {}
    for region, (atac_domains, h3_domains) in GROUPS.items():
        for assay, labels, xy, domains in (
            ("ATAC", atac_labels, atac_xy, atac_domains),
            ("H3K27ac", h3_labels, h3_aligned, h3_domains),
        ):
            selected = xy[labels.domain.isin(domains).to_numpy()]
            lo, hi = xy.min(axis=0), xy.max(axis=0)
            hist, y_edges, x_edges = np.histogram2d(
                selected[:, 1], selected[:, 0], bins=70,
                range=[(lo[1], hi[1]), (lo[0], hi[0])],
            )
            density = gaussian_filter(hist, 1.7)
            grids[region, assay] = (x_edges, y_edges, density / density.max())
    return grids


def plot_density_correspondence(matrix, grids):
    fig = plt.figure(figsize=(6.2, 3.45))
    heat = fig.add_axes([.115, .18, .39, .74])
    density_cosine_cmap = color_palette("rocket_r", as_cmap=True)
    artist = heat.pcolormesh(matrix, cmap=density_cosine_cmap, vmin=0, vmax=1, shading="flat")
    heat.set_xticks(np.arange(matrix.shape[1]) + .5, matrix.columns, fontsize=8)
    heat.set_yticks(np.arange(matrix.shape[0]) + .5, matrix.index, fontsize=8)
    heat.set(xlabel="H3K27ac domains", ylabel="ATAC domains", ylim=(13, 0))
    heat.tick_params(length=0, pad=2)
    for spine in heat.spines.values():
        spine.set_visible(False)
    cax = fig.add_axes([.53, .18, .016, .20])
    colorbar = fig.colorbar(artist, cax=cax, ticks=[0, 1])
    colorbar.solids.set_rasterized(False)
    cax.set_title("Density\ncosine", fontsize=8, pad=3)

    for row, (region, (x, y, width, height)) in enumerate(zip(REGIONS, BLOCKS)):
        for col, assay in enumerate(("ATAC", "H3K27ac")):
            x_edges, y_edges, values = grids[region, assay]
            ax = fig.add_axes([.67 + col * .15, .195 + (2 - row) * .245, .137, .225])
            surface = ax.pcolormesh(x_edges, y_edges, values, cmap="viridis", vmin=0, vmax=1,
                                    shading="flat", rasterized=False)
            ax.set_aspect("equal")
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set(color=DENSITY_MAP_FRAME_COLORS[region], linewidth=.8)
            if row == 0:
                ax.set_title(assay, fontsize=9, pad=3)
            if col == 0:
                ax.set_ylabel(region, fontsize=8, labelpad=3)
                fig.add_artist(ConnectionPatch(
                    xyA=(x + width, y + height / 2), coordsA=heat.transData,
                    xyB=(0, .5), coordsB=ax.transAxes, color="#333333", lw=.6,
                    linestyle=(0, (4, 4)),
                ))
        heat.add_patch(Rectangle((x, y), width, height, fill=False,
                                 edgecolor=MATRIX_OUTLINE_COLORS[region], lw=1.05))
    cax = fig.add_axes([.755, .085, .16, .017])
    colorbar = fig.colorbar(surface, cax=cax, orientation="horizontal", ticks=[0, 1])
    colorbar.solids.set_rasterized(False)
    cax.set_title("Normalized density", fontsize=8, pad=4)
    return fig


def plot_interval_support(support):
    values = support.pivot(index="atac_group", columns="h3_group", values="support_fraction").loc[REGIONS, REGIONS]
    fig = plt.figure(figsize=(4.15, 3.2))
    ax = fig.add_axes([.295, .23, .48, .62])
    vmax = max(.5, np.ceil(values.to_numpy().max() * 10) / 10)
    mesh = ax.pcolormesh(values, cmap="YlOrRd", vmin=0, vmax=vmax, shading="flat")
    ax.set(xlabel="H3K27ac region set", ylabel="Region-associated ATAC peaks",
           xlim=(0, 3), ylim=(3, 0), aspect="equal")
    ax.set_xticks(np.arange(3) + .5, REGIONS, rotation=28, ha="right", rotation_mode="anchor")
    ax.set_yticks(np.arange(3) + .5, REGIONS)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    for row, column in np.ndindex(values.shape):
        value = values.iloc[row, column]
        ax.text(column + .5, row + .5, f"{value:.2f}", ha="center", va="center",
                color="white" if value / vmax >= .83 else "#151515")
    cax = fig.add_axes([.815, .39, .025, .28])
    colorbar = fig.colorbar(mesh, cax=cax, ticks=[0, vmax])
    colorbar.solids.set_rasterized(False)
    cax.set_title("Ratio", fontsize=9, pad=4)
    return fig


def plot_cc_network(all_edges):
    """Lay out and draw all CC/oligo TF–target edges from the triplet calculation."""
    edges = all_edges.loc[all_edges.group.eq("CC/oligo")]
    tf_stats = (edges.groupby("TF", as_index=False)
                .agg(degree=("target_gene", "nunique"), support=("n_candidate_enhancers", "sum"))
                .sort_values(["degree", "support", "TF"], ascending=[False, False, True]))
    tf_order = tf_stats.TF.tolist()
    primary = (edges.sort_values(["target_gene", "n_candidate_enhancers", "max_P2G_r", "min_P2G_FDR", "TF"],
                                ascending=[True, False, False, True, True])
               .drop_duplicates("target_gene").set_index("target_gene").TF)
    graph = nx.Graph()
    for row in edges.itertuples(index=False):
        graph.add_edge(("TF", row.TF), ("target", row.target_gene),
                       weight=1 + np.log1p(row.n_candidate_enhancers))
    xy = nx.spring_layout(graph, seed=1234, k=1.55 / np.sqrt(len(graph)),
                          iterations=1200, weight="weight")
    tf_pairs = [(("TF", a), ("TF", b)) for i, a in enumerate(tf_order) for b in tf_order[i + 1:]]
    for _ in range(120):
        moved = False
        for left, right in tf_pairs:
            delta = xy[right] - xy[left]
            distance = np.linalg.norm(delta)
            if 1e-10 < distance < .105:
                shift = .5 * (.105 - distance) * delta / distance
                xy[left] -= shift
                xy[right] += shift
                moved = True
        if not moved:
            break

    fig, ax = plt.subplots(figsize=(5.35, 5.2))
    ax.set(title="TF–target regulatory network of CC/oligo", aspect="equal", xticks=[], yticks=[])
    ax.margins(.11)
    for row in edges.itertuples(index=False):
        start, end = xy["TF", row.TF], xy["target", row.target_gene]
        ax.plot([start[0], end[0]], [start[1], end[1]], color=TF_COLORS.get(row.TF, "#777777"),
                lw=.32 + .28 * min(row.n_candidate_enhancers, 3),
                alpha=.43 if primary[row.target_gene] == row.TF else .18,
                solid_capstyle="round", zorder=1)
    target_degrees = edges.groupby("target_gene").TF.nunique()
    target_xy = np.array([xy["target", gene] for gene in target_degrees.index])
    target_sizes = 13 + 3 * np.minimum(target_degrees, 4)
    ax.scatter(*target_xy.T, s=target_sizes,
               c="#E7A7CB", ec="white", lw=.35, zorder=3, label="Target gene")
    tf_labels = []
    tf_xy = []
    tf_sizes = []
    for tf, degree in tf_stats[["TF", "degree"]].itertuples(index=False, name=None):
        point = xy["TF", tf]
        size = 62 + 2.9 * np.sqrt(degree)
        tf_xy.append(point)
        tf_sizes.append(size)
        ax.scatter(*point, s=size, c=[TF_COLORS.get(tf, "#777777")],
                   ec="white", lw=.55, zorder=4)
        tf_labels.append(ax.text(*point, tf.upper(), fontsize=7.5, ha="center", va="center", zorder=5))

    target_handle = ax.collections[0]
    tf_handle = mpl.lines.Line2D([], [], marker="o", ls="none", color=TF_COLORS["Sox6"], ms=6)
    legend = ax.legend([tf_handle, target_handle], ["TF", "Target gene"], loc="lower left",
                       fontsize=8, frameon=False)
    fig.tight_layout()
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    occupied = [label.get_window_extent(renderer).expanded(1.05, 1.12) for label in [*tf_labels, legend]]
    points_to_pixels = renderer.points_to_pixels(1)
    marker_centers = ax.transData.transform(np.vstack((target_xy, tf_xy)))
    marker_radii = .5 * np.sqrt(np.r_[target_sizes, tf_sizes]) * points_to_pixels
    for gene in TARGET_LABELS:
        if gene not in primary.index:
            continue
        point = xy["target", gene]
        direction = point - xy["TF", primary[gene]]
        radial = 13 * direction / max(np.linalg.norm(direction), 1e-12)
        offsets = [radial, (11, 8), (-11, 8), (11, -8), (-11, -8),
                   (13, 0), (-13, 0), (13, 11), (-13, 11), (13, -11), (-13, -11)]
        center = ax.transData.transform(point)
        other = np.any(marker_centers != center, axis=1)
        relative_centers = marker_centers[other] - center
        other_radii = marker_radii[other]
        marker_radius = .5 * np.sqrt(target_sizes[gene])
        trial = ax.annotate(gene, point, textcoords="offset points", xytext=radial, fontsize=7.5, va="center")
        best = None
        for offset in offsets:
            trial.set_position(offset)
            alignment = "left" if offset[0] >= 0 else "right"
            trial.set_ha(alignment)
            box = trial.get_window_extent(renderer).expanded(1.05, 1.12)
            outside = not (ax.bbox.contains(box.x0, box.y0) and ax.bbox.contains(box.x1, box.y1))
            collisions = sum(box.overlaps(other_box) for other_box in occupied)
            # Score the straight segment and label against the other node markers.
            segment = np.asarray(offset) * points_to_pixels
            fraction = np.clip(relative_centers @ segment / (segment @ segment), 0, 1)
            crossing = np.linalg.norm(relative_centers - fraction[:, None] * segment, axis=1) < other_radii
            centers = marker_centers[other]
            nearest = np.clip(centers, (box.x0, box.y0), (box.x1, box.y1))
            covered = np.linalg.norm(centers - nearest, axis=1) < other_radii
            score = 100 * outside + 20 * collisions + 4 * crossing.sum() + covered.sum()
            if best is None or score < best[0]:
                best = score, offset, alignment, box
            if score == 0:
                break
        trial.remove()
        _, offset, alignment, box = best
        occupied.append(box)
        # End the leader at the boundary of the target marker.
        ax.annotate(gene, point, xytext=offset, textcoords="offset points", fontsize=7.5,
                    ha=alignment, va="center", zorder=6,
                    arrowprops={"arrowstyle": "-", "connectionstyle": "arc3,rad=0",
                                "color": "#555555", "lw": .45,
                                "shrinkA": 1.0, "shrinkB": marker_radius,
                                "relpos": (0 if alignment == "left" else 1, .5)})
    return fig
