"""Interactive native-coordinate plots with explicit, collision-safe exports.

Pyplot is imported only when rendering. The caller's Matplotlib backend is
preserved; these helpers never select a backend, save automatically, or close."""

import colorsys
import hashlib
import os
from pathlib import Path
import re
import tempfile
import numpy as np
import pandas as pd


DOMAIN_COLORS = {
    "D0": "#56B4E9", "D1": "#E69F00", "D2": "#F0E442", "D3": "#76896F",
    "D4": "#CAB2D6", "D5": "#D0B8A8", "D6": "#40BFC1", "D7": "#468966",
    "D8": "#999999", "D9": "#99D98C", "D10": "#BDBDBD", "D11": "#E41A1C",
    "D12": "#CC79A7",
}


def _categories(labels):
    categories = list(dict.fromkeys(map(str, labels)))
    if all(re.fullmatch(r"D\d+", value) for value in categories):
        categories.sort(key=lambda value: int(value[1:]))
    return categories


def category_palette(labels, palette=None):
    """Select explicit colors, or the default colors for native domain IDs."""
    categories = _categories(labels)
    if palette is not None:
        missing = set(categories) - set(palette)
        if missing:
            raise ValueError(f"palette has no color for {', '.join(sorted(missing))}")
        from matplotlib.colors import to_rgba
        selected = {label: color for label, color in palette.items() if label in categories}
        for color in selected.values():
            to_rgba(color)
        return selected
    result = {}
    for value in categories:
        if value in DOMAIN_COLORS:
            result[value] = DOMAIN_COLORS[value]
        else:
            digest = hashlib.sha256(value.encode()).digest()
            hue = int.from_bytes(digest[:8], "big") / 2**64
            result[value] = colorsys.hsv_to_rgb(hue, .60, .78)
    return result


def _labels(values):
    if hasattr(values, "labels"):
        values = values.labels
    if isinstance(values, pd.DataFrame):
        if "domain" not in values:
            raise ValueError("labels DataFrame requires domain column")
        values = values.domain
    result = np.asarray(values, dtype=str)
    if result.ndim != 1 or len(result) == 0:
        raise ValueError("nonempty one-dimensional labels required")
    return result


def _coordinates(values):
    if isinstance(values, pd.DataFrame):
        columns = ("x", "y") if {"x", "y"} <= set(values) else ("UMAP1", "UMAP2")
        values = values[list(columns)].to_numpy()
    result = np.asarray(values, dtype=float)
    if result.ndim != 2 or result.shape[1] < 2 or not np.isfinite(result).all():
        raise ValueError("finite two-dimensional coordinates required")
    return result


def _named_ids(values):
    if hasattr(values, "labels"):
        values = values.labels
    if not isinstance(values, (pd.DataFrame, pd.Series)):
        return None
    def text(value):
        value = value.decode() if isinstance(value, bytes) else str(value)
        if not value or value in ("nan", "None", "<NA>"):
            raise ValueError("plot IDs must be nonempty text")
        return value
    if isinstance(values, pd.DataFrame) and {"entity_type", "entity_id"} <= set(values):
        keys = [(text(kind), text(name)) for kind, name in zip(values.entity_type, values.entity_id)]
        kind = "typed"
    elif isinstance(values, pd.DataFrame) and "spot_id" in values:
        keys, kind = [text(value) for value in values.spot_id], "untyped"
    elif isinstance(values, pd.DataFrame) and "entity_id" in values:
        keys, kind = [text(value) for value in values.entity_id], "untyped"
    elif (values.index.name in ("spot_id", "entity_id")
          or not values.index.equals(pd.RangeIndex(len(values)))):
        keys, kind = [text(value) for value in values.index], "untyped"
    else:
        return None
    if len(set(keys)) != len(keys):
        raise ValueError("plot cohort IDs must be unique")
    return kind, keys


def _aligned_labels(coordinates, labels):
    values = _labels(labels)
    left, right = _named_ids(coordinates), _named_ids(labels)
    if left is not None and right is not None:
        if left[0] != right[0] or set(left[1]) != set(right[1]):
            raise ValueError("plot coordinate and label cohort IDs differ")
        lookup = {key: value for key, value in zip(right[1], values)}
        values = np.array([lookup[key] for key in left[1]])
    return values


def _legend(ax, palette, marker="o"):
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], color=color, marker=marker, linestyle="None",
                      markersize=4.5, label=label) for label, color in palette.items()]
    ax.legend(handles=handles, title="Domains", loc="center left", bbox_to_anchor=(1.005, .5),
              fontsize=6.2, title_fontsize=7, frameon=False,
              handletextpad=.4, labelspacing=.35)


def spatial_domains(coordinates, labels, title="", ax=None, point_size=20,
                    annotate_domains=False, *, legend=True, palette=None, marker="o"):
    """Plot native increasing-y coordinates, returning (Figure, Axes).

    Default panel size is 3.3 x 3.35 inches and scatter area is 20 pt². Domain
    legends are independent from the optional centroid text annotations.
    palette optionally maps every displayed domain name to an explicit color.
    """
    labels = _aligned_labels(coordinates, labels)
    coordinates = _coordinates(coordinates)
    if len(coordinates) != len(labels):
        raise ValueError("coordinates and labels differ in length")
    if not np.isfinite(point_size) or point_size <= 0:
        raise ValueError("point_size must be positive and finite")
    from matplotlib import pyplot as plt
    if ax is None:
        fig, ax = plt.subplots(figsize=(3.3, 3.35), constrained_layout=True)
    else:
        fig = ax.figure
    ax.yaxis.grid(False)
    palette = category_palette(labels, palette)
    ax.scatter(coordinates[:, 0], coordinates[:, 1], c=[palette[x] for x in labels],
               s=point_size, linewidths=0, marker=marker)
    ax.set_aspect("equal")
    ax.yaxis.set_inverted(False)
    ax.set_title(str(title), fontsize=9)
    ax.set_axis_off()
    if annotate_domains:
        for label in palette:
            center = coordinates[labels == label, :2].mean(axis=0)
            ax.text(*center, label, ha="center", va="center")
    if legend:
        _legend(ax, palette, marker)
    return fig, ax


def spatial_panel(cases, figsize=(13.2, 3.35), *, palette=None, point_size=20, marker="o"):
    """Render a row of {coordinates, labels, title} cases with one shared legend."""
    cases = list(cases)
    if not cases:
        raise ValueError("at least one spatial case is required")
    # Validate before allocating a figure.
    prepared = [(_coordinates(case["coordinates"]), _aligned_labels(case["coordinates"], case["labels"]),
                 case.get("title", "")) for case in cases]
    if any(len(coords) != len(labels) for coords, labels, _ in prepared):
        raise ValueError("coordinates and labels differ in length")
    palette = category_palette(np.concatenate([x[1] for x in prepared]), palette)
    from matplotlib import pyplot as plt
    fig, axes = plt.subplots(1, len(cases), figsize=figsize, squeeze=False, constrained_layout=True)
    for ax, (coords, labels, title) in zip(axes[0], prepared):
        spatial_domains(coords, labels, title=title, ax=ax, legend=False,
                        palette=palette, point_size=point_size, marker=marker)
    _legend(axes[0, -1], palette, marker)
    return fig, axes[0]


def embeddings(table, labels=None, title="", ax=None, point_size=20):
    """Scatter explicit embedding coordinates or an EmbeddingResult table."""
    if hasattr(table, "table"):
        table = table.table
    coordinates = _coordinates(table)
    if labels is None:
        labels = (table.entity_type.to_numpy() if isinstance(table, pd.DataFrame) and "entity_type" in table
                  else np.repeat("spot", len(coordinates)))
    labels = _aligned_labels(table, labels)
    if len(labels) != len(coordinates):
        raise ValueError("embedding coordinates and labels differ in length")
    if not np.isfinite(point_size) or point_size <= 0:
        raise ValueError("point_size must be positive and finite")
    from matplotlib import pyplot as plt
    if ax is None:
        fig, ax = plt.subplots(figsize=(3.3, 3.35), constrained_layout=True)
    else:
        fig = ax.figure
    for label, color in category_palette(labels).items():
        subset = labels == label
        ax.scatter(coordinates[subset, 0], coordinates[subset, 1], color=color,
                   s=point_size, linewidths=0, alpha=.65, label=label)
    ax.set_xlabel("UMAP1")
    ax.set_ylabel("UMAP2")
    ax.yaxis.set_inverted(False)
    ax.set_title(str(title))
    ax.legend(frameon=False, fontsize=7)
    return fig, ax


def _umap_panel_axes(ax, title, aspect):
    """Apply the reference panel frame locally, without changing rcParams."""
    ax.set_aspect(aspect)
    ax.yaxis.grid(False)
    ax.yaxis.set_inverted(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("")
    ax.set_ylabel("")
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(.5)
    ax.set_title(str(title), pad=5, fontsize=8, fontfamily="Liberation Sans")




def spot_umap(table, labels=None, title="SEAREG (spots)", ax=None, point_size=2, *, palette=None):
    """Plot all spot coordinates on the tutorial's 75 x 48.35 mm canvas.

    A supplied Axes keeps its original figure size and position. Use
    save_figure(..., bbox_inches=None) to retain the full physical canvas.
    """
    if hasattr(table, "table"):
        table = table.table
    coordinates = _coordinates(table)
    if not len(coordinates):
        raise ValueError("nonempty spot UMAP coordinates required")
    if not np.isfinite(point_size) or point_size <= 0:
        raise ValueError("point_size must be positive and finite")
    show_legend = labels is not None
    if show_legend:
        labels = _joint_spot_labels(table, np.ones(len(coordinates), dtype=bool), labels)
        palette = category_palette(labels, palette)
    else:
        labels = np.repeat("spot", len(coordinates))
        palette = {"spot": DOMAIN_COLORS["D0"]}
    from matplotlib import pyplot as plt
    from matplotlib.lines import Line2D
    if ax is None:
        fig = plt.figure(figsize=(75 / 25.4, 48.35 / 25.4), dpi=300)
        ax = fig.add_axes([.025, .055, .60, .86])
    else:
        fig = ax.figure
    for label in dict.fromkeys(labels):
        subset = labels == label
        ax.scatter(coordinates[subset, 0], coordinates[subset, 1], color=palette[label],
                   s=point_size, linewidths=0, label=label)
    _umap_panel_axes(ax, title, "auto")
    if show_legend:
        handles = [Line2D([], [], marker="o", lw=0, ms=4, color=color, label=label)
                   for label, color in palette.items()]
        ax.legend(handles=handles, ncol=2, loc="center left", bbox_to_anchor=(1.02, .5),
                  fontsize=7, frameon=False, handlelength=.7,
                  handletextpad=.4, labelspacing=.25, borderaxespad=0,
                  prop={"family": "Liberation Sans", "size": 7})
    return fig, ax


def _joint_spot_labels(table, spot_mask, labels):
    """Align labels to spot rows without merging IDs across entity types."""
    if hasattr(labels, "labels"):
        labels = labels.labels
    if isinstance(labels, pd.DataFrame):
        if "domain" not in labels:
            raise ValueError("labels DataFrame requires domain column")
        raw = labels.domain.to_numpy()
        if "entity_type" in labels:
            if not labels.entity_type.eq("spot").all():
                raise ValueError("joint UMAP domain labels must describe spots only")
            labels = labels.drop(columns="entity_type")
    else:
        raw = np.asarray(labels, dtype=object)
    if raw.ndim != 1 or len(raw) != int(spot_mask.sum()):
        raise ValueError("joint UMAP labels must match the spot population")
    if (not len(raw) or pd.isna(raw).any()
            or any(str(value).strip() in ("", "nan", "None", "<NA>") for value in raw)):
        raise ValueError("joint UMAP spot labels must be nonempty and complete")
    # A non-default pandas index carries identity, even when its name is absent.
    if isinstance(labels, (pd.DataFrame, pd.Series)) and _named_ids(labels) is None:
        if not isinstance(labels.index, pd.RangeIndex):
            labels = labels.rename_axis("spot_id")
    if isinstance(table, pd.DataFrame):
        spots = table.loc[spot_mask].drop(columns="entity_type", errors="ignore")
    else:
        spots = np.asarray(table)[spot_mask]
    if _named_ids(labels) is not None and _named_ids(spots) is None:
        raise ValueError("named spot labels require named spot coordinates")
    return _aligned_labels(spots, labels)


def _joint_annotations(table, annotations):
    """Resolve explicit feature labels against typed IDs in the current table."""
    if annotations is None:
        return None
    required = {"entity_type", "entity_id", "display_domain"}
    if (not isinstance(annotations, pd.DataFrame) or annotations.empty
            or not required <= set(annotations)):
        raise ValueError("annotations require entity_type/entity_id/display_domain")
    if not isinstance(table, pd.DataFrame) or not {"entity_type", "entity_id"} <= set(table):
        raise ValueError("annotations require typed coordinate IDs")
    if not annotations.entity_type.isin(["peak", "motif", "kmer"]).all():
        raise ValueError("annotations must identify peak, motif or kmer nodes")
    if (annotations.display_domain.isna().any()
            or annotations.display_domain.astype(str).str.strip().eq("").any()):
        raise ValueError("annotations require a nonempty display_domain")
    try:
        _named_ids(annotations)
    except ValueError as exc:
        raise ValueError("annotation IDs must be nonempty and unique") from exc
    selected = annotations[["entity_type", "entity_id", "display_domain"]].merge(
        table[["entity_type", "entity_id", "UMAP1", "UMAP2"]],
        on=["entity_type", "entity_id"], how="left", validate="one_to_one",
    )
    if selected[["UMAP1", "UMAP2"]].isna().any().any():
        raise ValueError("annotation IDs are missing from the joint coordinate table")
    return selected


def _draw_joint_annotations(ax, selected, text_expand, max_leader_mm):
    """Bind labels to current typed nodes and choose nearby text positions."""
    from matplotlib.patches import FancyArrowPatch
    from ._label_layout import place_labels
    palette = category_palette(selected.display_domain)
    texts = []
    for row in selected.itertuples(index=False):
        label = (str(row.entity_id).split("_MA")[0] if row.entity_type == "motif"
                 else str(row.entity_id).replace("chr", "Chr", 1))
        texts.append(ax.text(row.UMAP1, row.UMAP2, label, fontsize=6,
                             ha="center", va="center",
                             color=palette[str(row.display_domain)],
                             fontfamily="Liberation Sans"))
    anchors = selected[["UMAP1", "UMAP2"]].to_numpy()
    place_labels(ax, texts, anchors, expand=text_expand, max_leader_mm=max_leader_mm)
    for text, anchor in zip(texts, anchors):
        ax.add_patch(FancyArrowPatch(
            text.get_position(), anchor, patchA=text, transform=ax.transData,
            arrowstyle="-", lw=.6, color=text.get_color(), shrinkA=0, shrinkB=1,
        ))


def joint_umap(table, labels=None, title="SEAREG (spots + peaks + motifs)", ax=None,
               point_size=5.2, *, peak_size=2.4, motif_size=4.2, annotations=None,
               text_expand=(1.05, 1.2), text_max_leader_mm=6, palette=None):
    """Layer all native joint coordinates as peaks, DNA features and spots.

    Peak triangles and motif/kmer squares sit below the spot circles. Optional
    domain labels describe only the spot rows: named pandas labels are aligned
    by spot ID, while one-dimensional arrays follow the table's spot-row order.
    The 75 x 113.2 mm canvas and per-type areas follow the requested tutorial.
    motif_size also applies to kmer nodes. A supplied Axes keeps its canvas and
    position. The function never fits, filters, samples, saves, or changes
    coordinates. Explicit annotations add only text and leaders; they do not
    redraw, enlarge or recolor selected points. Text starts at the current
    node, then nearby non-overlapping text positions are selected with
    bounded, visible leaders. text_expand pads text boxes for collision checks;
    text_max_leader_mm bounds the visible leader length (default 6 mm).
    No saved label positions or offsets are used. Export at 300 dpi with
    bbox_inches=None to match the tutorial's saved image dimensions.
    """
    if hasattr(table, "table"):
        table = table.table
    coordinates = _coordinates(table)
    if len(coordinates) == 0:
        raise ValueError("nonempty joint UMAP coordinates required")
    for name, size in (("point_size", point_size), ("peak_size", peak_size), ("motif_size", motif_size)):
        if not np.isfinite(size) or size <= 0:
            raise ValueError(f"{name} must be positive and finite")
    if isinstance(table, pd.DataFrame) and "entity_type" in table:
        kinds = table.entity_type.to_numpy(dtype=str)
        if not set(kinds) <= {"spot", "peak", "motif", "kmer"}:
            raise ValueError("joint UMAP entity types must be spot, peak, motif or kmer")
    else:
        kinds = np.repeat("spot", len(coordinates))
    _named_ids(table)  # Validate typed uniqueness without conflating same-name IDs.
    selected = _joint_annotations(table, annotations)
    expand = np.asarray(text_expand, dtype=float)
    if expand.shape != (2,) or not np.isfinite(expand).all() or (expand < 1).any():
        raise ValueError("text_expand requires two finite multipliers of at least one")
    if not np.isfinite(text_max_leader_mm) or text_max_leader_mm <= 0:
        raise ValueError("text_max_leader_mm must be positive and finite")
    spot_mask = kinds == "spot"
    if labels is not None:
        aligned = _joint_spot_labels(table, spot_mask, labels)
        palette = category_palette(aligned, palette)
        spot_colors = [palette[value] for value in aligned]
    else:
        spot_colors = DOMAIN_COLORS["D0"]

    from matplotlib import pyplot as plt
    from matplotlib.lines import Line2D
    if ax is None:
        fig = plt.figure(figsize=(75 / 25.4, 113.2 / 25.4), dpi=300)
        ax = fig.add_axes([.055, .13, .89, .75])
    else:
        fig = ax.figure
    styles = (("peak", "^", "#CFCFCF", peak_size, .22, 1),
              ("motif", "s", "#2C2C2C", motif_size, .20, 1),
              ("kmer", "s", "#2C2C2C", motif_size, .20, 1),
              ("spot", "o", spot_colors, point_size, .65, 3))
    for kind, marker, color, size, alpha, zorder in styles:
        subset = kinds == kind
        if not subset.any():
            continue
        ax.scatter(coordinates[subset, 0], coordinates[subset, 1], c=color,
                   marker=marker, s=size, linewidths=0, alpha=alpha,
                   zorder=zorder, label=kind)
    _umap_panel_axes(ax, title, "equal")
    ax.set_title(str(title), pad=4, fontsize=8, fontfamily="Liberation Sans")
    legend_styles = (("spot", "o", "#888888"), ("peak", "^", "#BDBDBD"),
                     ("motif", "s", "#333333"), ("kmer", "s", "#333333"))
    handles = [Line2D([], [], marker=marker, lw=0, ms=4, color=color, label=kind.title())
               for kind, marker, color in legend_styles if np.any(kinds == kind)]
    ax.legend(handles=handles, loc="upper right", frameon=False, fontsize=7,
              handlelength=.8, handletextpad=.4, labelspacing=.25,
              prop={"family": "Liberation Sans", "size": 7})
    if selected is not None:
        _draw_joint_annotations(ax, selected, tuple(expand), text_max_leader_mm)
    return fig, ax


def save_figure(fig, stem, dpi=450, formats=("png", "pdf", "svg"), *, bbox_inches="tight"):
    """Explicitly export native PDF/SVG plus PNG without replacing any target.

    stem is a literal file prefix: stem='map.v1' creates map.v1.png etc. Every
    requested target is checked before rendering. Publication is exclusive and
    exports created by this call are removed if publication fails partway.
    bbox_inches='tight' preserves the existing cropped-export default. Pass None
    to keep the exact figure canvas, including physical PDF and PNG dimensions.
    The figure remains open and editable after saving.
    """
    formats = tuple(formats)
    if not formats or len(set(formats)) != len(formats) or any(x not in ("png", "pdf", "svg") for x in formats):
        raise ValueError("formats must be distinct PNG/PDF/SVG extensions in lowercase")
    if isinstance(dpi, bool) or not np.isfinite(dpi) or dpi <= 0:
        raise ValueError("dpi must be positive and finite")
    if bbox_inches is not None and not (isinstance(bbox_inches, str) and bbox_inches == "tight"):
        raise ValueError("bbox_inches must be None or 'tight'")
    stem = Path(stem)
    targets = tuple(Path(str(stem) + "." + extension) for extension in formats)
    for target in targets:
        if target.exists() or target.is_symlink():
            raise FileExistsError(target)
    from .utils import protect_output
    protect_output(stem.parent)
    stem.parent.mkdir(parents=True, exist_ok=True)
    from matplotlib import rc_context
    published = []
    try:
        with tempfile.TemporaryDirectory(prefix="." + stem.name + ".export-", dir=stem.parent) as directory:
            staged = []
            with rc_context({"pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
                             "savefig.bbox": bbox_inches}):
                for extension in formats:
                    path = Path(directory) / ("figure." + extension)
                    fig.savefig(path, format=extension, dpi=dpi, bbox_inches=bbox_inches)
                    staged.append(path)
            for source, target in zip(staged, targets):
                os.link(source, target)  # Atomic, exclusive same-filesystem publication.
                published.append(target)
    except BaseException:
        for target in published:
            target.unlink(missing_ok=True)
        raise
    return targets


def render_analysis(analysis,output):
    """Render saved numerical results without PCA, neighbors, clustering or UMAP."""
    from .analysis import load_result
    from .artifacts import StageArtifact,stage_metadata
    from .utils import artifact_writer,write_json
    from ._analysis_io import _figure_metadata,DISPLAY
    from matplotlib import pyplot as plt
    result=load_result(analysis.path if hasattr(analysis,'path') else analysis)
    upstream=StageArtifact.load(result.path)
    metadata=stage_metadata(output,upstreams={'analysis':upstream},protocol='saved_analysis_render_v1',
                            display=DISPLAY,figures={})
    config=result.metadata['config'];first=None
    with artifact_writer(output,'analysis_figures',metadata) as stage:
        if config['domains']:
            fixed=config.get('fixed_resolution') is not None
            for k in ((None,) if fixed else config['target_domains']):
                domain=result.domains(k);labels=domain.labels
                if first is None:first=labels
                slices=labels.slice_id.to_numpy();unique=list(dict.fromkeys(slices.tolist()))
                if len(unique)==1:
                    fig,_=spatial_domains(domain.coordinates,labels,title=unique[0])
                else:
                    cases=[{'coordinates':domain.coordinates[slices==s],
                        'labels':labels.loc[slices==s],'title':s} for s in unique]
                    fig,_=spatial_panel(cases,figsize=(3.3*len(unique),3.35))
                stem='fixed_resolution' if fixed else f'K{k}'
                save_figure(fig,stage/f'spatial_domains_{stem}')
                metadata['figures'][f'spatial_domains_{stem}']=_figure_metadata(fig);plt.close(fig)
        if config['spot_umap']:
            fig,_=spot_umap(result.spot_embedding().table,labels=first)
            save_figure(fig,stage/'spot_umap',dpi=300,bbox_inches=None)
            metadata['figures']['spot_umap']=_figure_metadata(fig,export_dpi=300);plt.close(fig)
        if config['joint']:
            fig,_=joint_umap(result.joint_embedding().table,labels=first)
            save_figure(fig,stage/'joint_umap',dpi=300,bbox_inches=None)
            metadata['figures']['joint_umap']=_figure_metadata(fig,export_dpi=300);plt.close(fig)
        write_json(stage/'render.json',metadata)
    return StageArtifact.load(output)
