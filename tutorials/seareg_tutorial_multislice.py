"""Annotation joins and color-only views for the embryo multislice tutorial.

Reference annotations enter only these display helpers. Native predicted domain
IDs, numerical coordinates, model inputs and fit procedures are never changed.
"""

import colorsys
import hashlib
import json
from numbers import Real
from pathlib import Path
import re

import numpy as np
import pandas as pd


UNKNOWN_COLOR = "#999999"


def _text(values, name, *, unique=False):
    values = np.asarray(values, dtype=object)
    if values.ndim != 1 or not len(values) or pd.isna(values).any():
        raise ValueError(f"{name} must be nonempty one-dimensional text")
    result = np.array([str(value) for value in values])
    if any(not value.strip() for value in result):
        raise ValueError(f"{name} must be nonempty text")
    if unique and len(set(result)) != len(result):
        raise ValueError(f"duplicate {name}")
    return result


def _annotations(values):
    if isinstance(values, pd.DataFrame):
        values = values.annotation
    values = np.asarray(values, dtype=object)
    if values.ndim != 1:
        raise ValueError("annotations must be one-dimensional")
    return np.array(["Unknown" if pd.isna(value) or not str(value).strip()
                     else str(value) for value in values])


def load_annotations(path, spot_ids, slice_ids=None, annotation_column="annotation"):
    """Join a TSV to ordered spot IDs, filling absent labels with Unknown.

    Duplicate/extra spot IDs and conflicting supplied sample IDs are errors.
    The returned columns are spot_id, sample_id, annotation; no spot is dropped.
    """
    ids = _text(spot_ids, "spot IDs", unique=True)
    table = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    if not {"spot_id", annotation_column} <= set(table):
        raise ValueError("annotation table requires spot_id and annotation columns")
    reference_ids = (_text(table.spot_id, "annotation spot IDs", unique=True)
                     if len(table) else np.array([], dtype=str))
    if set(reference_ids) - set(ids):
        raise ValueError("extra annotation spot IDs outside the counts cohort")
    ordered = table.set_index("spot_id").reindex(ids)
    if slice_ids is not None:
        slices = _text(slice_ids, "sample IDs")
        if len(slices) != len(ids):
            raise ValueError("sample IDs and spot IDs differ in length")
        if "sample_id" in ordered:
            present = ordered.sample_id.notna()
            if (ordered.loc[present, "sample_id"].to_numpy() != slices[present]).any():
                raise ValueError("annotation sample IDs differ from the counts cohort")
    else:
        slices = (_annotations(ordered.sample_id) if "sample_id" in ordered
                  else np.repeat("Unknown", len(ids)))
    return pd.DataFrame({"spot_id": ids, "sample_id": slices,
                         "annotation": _annotations(ordered[annotation_column])})


def annotation_palette(annotations, palette_path=None, *, unknown_color=UNKNOWN_COLOR):
    """Return one cohort palette, with an explicit display color for Unknown.

    palette_path optionally names a JSON object mapping annotation names to
    Matplotlib colors. Without a file, stable native category colors are used.
    The default Unknown gray stays compatible with existing tutorials; a named
    reference palette can supply its official gray through ``unknown_color``.
    """
    from seareg.plotting import category_palette

    names = list(dict.fromkeys(_annotations(annotations)))
    if "Unknown" not in names:
        names.append("Unknown")
    supplied = None
    if palette_path is not None:
        supplied = json.loads(Path(palette_path).read_text(encoding="utf-8"))
        if not isinstance(supplied, dict):
            raise ValueError("annotation palette must be a JSON object")
        supplied = dict(supplied, Unknown=unknown_color)
    palette = category_palette(names, supplied)
    palette["Unknown"] = unknown_color
    return palette


def _domain_order(values):
    def key(value):
        return (0, int(value[1:])) if re.fullmatch(r"D\d+", value) else (1, value)
    return sorted(set(values), key=key)


def _membership_digest(ids):
    encoded = json.dumps(sorted(map(str, ids)), ensure_ascii=True,
                         separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def load_approved_domain_palette(labels, palette_path):
    """Load selected-example colors only for the exact saved spot partition.

    Membership hashes make this independent of row order and native cluster-ID
    permutations. Return ``None`` for a different cohort or partition, so new
    training/user data can use ordinary display matching. Malformed palette
    files raise ``ValueError``. Input labels are never renamed or mutated.
    """
    from matplotlib.colors import to_rgba, to_hex

    asset = json.loads(Path(palette_path).read_text(encoding="utf-8"))
    if not isinstance(asset, dict):
        raise ValueError("invalid approved domain palette schema")
    required = {"display_domain", "display_label", "color", "spots", "members_sha256"}
    entries = asset.get("domains", [])
    if (asset.get("schema_version") != 1 or not isinstance(entries, list) or not entries
            or any(not isinstance(item, dict) or not required <= set(item) for item in entries)):
        raise ValueError("invalid approved domain palette schema")
    aliases = _text([item["display_domain"] for item in entries], "approved display aliases", unique=True)
    digests = _text([item["members_sha256"] for item in entries], "approved membership hashes", unique=True)
    _text([item["display_label"] for item in entries], "approved display labels")
    try:
        color_ids = [to_hex(to_rgba(item["color"]), keep_alpha=True) for item in entries]
    except (ValueError, TypeError) as error:
        raise ValueError("invalid approved palette color") from error
    if len(set(color_ids)) != len(color_ids):
        raise ValueError("approved palette colors must be distinct")
    if hasattr(labels, "labels"):
        labels = labels.labels
    if not isinstance(labels, pd.DataFrame) or not {"spot_id", "domain"} <= set(labels):
        raise ValueError("domain labels require spot_id and domain columns")
    ids = _text(labels.spot_id, "domain spot IDs", unique=True)
    values = _text(labels.domain, "domain labels")
    if len(ids) != asset.get("spot_count") or _membership_digest(ids) != asset.get("spot_ids_sha256"):
        return None
    observed = {_membership_digest(ids[values == domain]): (domain, int(np.count_nonzero(values == domain)))
                for domain in _domain_order(values)}
    if set(observed) != set(digests):
        return None
    colors, records = {}, []
    for alias, item in zip(aliases, entries):
        native, count = observed[item["members_sha256"]]
        if count != item["spots"]:
            return None
        colors[native] = item["color"]
        records.append({"domain": native, "display_domain": alias,
                        "display_label": item["display_label"], "color": item["color"],
                        "domain_spots": count, "strategy": "approved_selected"})
    return colors, pd.DataFrame(records)


def match_domain_palette(labels, annotations, palette, *, strategy="one_to_one",
                         min_match_fraction=0.0):
    """Match native domain display colors to known anatomy by spot identity.

    Both tables must carry spot_id; domain labels are joined by identity, never
    row order. The default ``one_to_one`` uses maximum-total-overlap Hungarian
    assignment, with distinct fallback colors for unmatched domains. A positive
    ``min_match_fraction`` excludes candidate pairs below that fraction of the
    domain's known annotations before assignment; unmatched domains remain free
    to use fallback colors. The objective remains total raw overlap. ``dominant``
    selects each domain's most frequent known annotation independently, allowing
    several domains to share that exact reference color. Ties use sorted names;
    domains without known annotations receive the palette's Unknown gray.
    Unknown never competes with known labels. Return (domain_palette,
    mapping_table), leaving native IDs and both input tables unchanged.
    Mapping ``majority_*`` columns describe each domain's largest known overlap
    independently of eligibility or the final assigned reference.
    """
    from matplotlib.colors import to_hex, to_rgba
    from scipy.optimize import linear_sum_assignment

    if strategy not in ("one_to_one", "dominant"):
        raise ValueError("strategy must be one_to_one or dominant")
    if (isinstance(min_match_fraction, (bool, np.bool_))
            or not isinstance(min_match_fraction, Real)
            or not 0 <= min_match_fraction <= 1):
        raise ValueError("min_match_fraction must be a finite real number in [0, 1]")
    min_match_fraction = float(min_match_fraction)
    if strategy == "dominant" and min_match_fraction > 0:
        raise ValueError("positive min_match_fraction requires strategy='one_to_one'")
    if hasattr(labels, "labels"):
        labels = labels.labels
    if not isinstance(labels, pd.DataFrame) or not {"spot_id", "domain"} <= set(labels):
        raise ValueError("domain labels require spot_id and domain columns")
    if not isinstance(annotations, pd.DataFrame) or not {"spot_id", "annotation"} <= set(annotations):
        raise ValueError("annotations require spot_id and annotation columns")
    ids = _text(labels.spot_id, "domain spot IDs", unique=True)
    values = _text(labels.domain, "domain labels")
    reference_ids = _text(annotations.spot_id, "annotation spot IDs", unique=True)
    if set(reference_ids) - set(ids):
        raise ValueError("extra annotation spot IDs outside the domain cohort")
    anatomy = _annotations(annotations.set_index("spot_id").reindex(ids).annotation)
    known = sorted(set(anatomy) - {"Unknown"})
    missing = set(known) - set(palette)
    if missing:
        raise ValueError(f"palette has no color for {', '.join(sorted(missing))}")
    used = {to_hex(to_rgba(color), keep_alpha=True) for color in palette.values()}
    domains = _domain_order(values)
    overlap = np.zeros((len(domains), len(known)), dtype=np.int64)
    for i, domain in enumerate(domains):
        for j, annotation in enumerate(known):
            overlap[i, j] = np.count_nonzero((values == domain) & (anatomy == annotation))
    matches = {}
    if known:
        if strategy == "dominant":
            matches = {i: int(np.argmax(row)) for i, row in enumerate(overlap) if row.max() > 0}
        elif min_match_fraction > 0:
            known_counts = overlap.sum(axis=1, keepdims=True)
            fractions = np.divide(overlap, known_counts, out=np.zeros_like(overlap, dtype=float),
                                  where=known_counts > 0)
            eligible = (overlap > 0) & (fractions >= min_match_fraction)
            # Each domain has a private zero-cost unmatched option. Ineligible
            # real pairs cannot reserve colors needed by another domain.
            costs = np.full((len(domains), len(known) + len(domains)), np.inf)
            costs[:, :len(known)] = np.where(eligible, -overlap.astype(float), np.inf)
            costs[np.arange(len(domains)), len(known) + np.arange(len(domains))] = 0.
            rows, columns = linear_sum_assignment(costs)
            matches = {int(row): int(column) for row, column in zip(rows, columns)
                       if column < len(known)}
        else:
            rows, columns = linear_sum_assignment(-overlap)
            matches = {int(row): int(column) for row, column in zip(rows, columns)
                       if overlap[row, column] > 0}
    colors, records = {}, []
    fallback_index = 0
    for i, domain in enumerate(domains):
        matched = i in matches
        annotation = known[matches[i]] if matched else ("Unknown" if strategy == "dominant" else "")
        count = int(overlap[i, matches[i]]) if matched else 0
        if matched:
            color = palette[annotation]
        elif strategy == "dominant":
            color = palette.get("Unknown", UNKNOWN_COLOR)
        else:
            while True:
                hue = (fallback_index * .618033988749895 + .07) % 1
                fallback_index += 1
                color = to_hex(colorsys.hsv_to_rgb(hue, .62, .78))
                identity = to_hex(color, keep_alpha=True)
                if identity not in used:
                    used.add(identity)
                    break
        colors[domain] = color
        known_count = int(np.count_nonzero((values == domain) & (anatomy != "Unknown")))
        majority_index = int(np.argmax(overlap[i])) if known_count else None
        majority_count = int(overlap[i, majority_index]) if known_count else 0
        records.append({"domain": domain, "annotation": annotation, "overlap": count,
                        "domain_spots": int(np.count_nonzero(values == domain)),
                        "known_spots": known_count, "matched": matched, "color": color,
                        "match_fraction": count / known_count if known_count else 0.,
                        "majority_reference": known[majority_index] if known_count else "",
                        "majority_spots": majority_count,
                        "majority_fraction": majority_count / known_count if known_count else 0.,
                        "strategy": strategy, "min_match_fraction": min_match_fraction})
    return colors, pd.DataFrame(records)


def umap_panels(table, views, *, joint=False, point_size=4, peak_size=1.5):
    """Show section/domain/reference colors on three copies of saved UMAPs.

    Each view is (title, named spot-label table, palette, legend title).
    Joint views keep all peak coordinates gray and share one entity legend.
    This helper only plots; coordinate fitting and annotation joins stay separate.
    """
    from matplotlib import pyplot as plt
    from matplotlib.lines import Line2D
    from seareg import plotting

    views = list(views)
    if len(views) != 3:
        raise ValueError("UMAP comparison requires three views")
    fig = plt.figure(figsize=(15, 3.6), dpi=150)
    axes = [fig.add_axes([left, .16, .185, .77]) for left in (.04, .37, .70)]
    entity_handles = []
    for ax, (title, labels, palette, legend_title) in zip(axes, views):
        if labels is None:
            ax.set_axis_off()
            ax.text(.5, .5, "Spatial domains not requested", transform=ax.transAxes,
                    ha="center", va="center", fontsize=8)
            continue
        if joint:
            plotting.joint_umap(table, labels=labels, palette=palette, title=title,
                                ax=ax, point_size=point_size, peak_size=peak_size)
        else:
            plotting.spot_umap(table, labels=labels, palette=palette, title=title,
                               ax=ax, point_size=point_size)
        previous = ax.get_legend()
        if previous is not None:
            if joint and not entity_handles:
                entity_handles = list(previous.legend_handles)
            previous.remove()
        selected = plotting.category_palette(labels["domain"], palette)
        handles = [Line2D([], [], marker="o", lw=0, markersize=4,
                          color=color, label=str(label)) for label, color in selected.items()]
        ax.legend(handles=handles, title=legend_title, loc="center left",
                  bbox_to_anchor=(1.025, .5), ncol=1, fontsize=8, title_fontsize=9,
                  frameon=False, handlelength=.8, handletextpad=.4,
                  labelspacing=.3, borderaxespad=0)
        ax.set_title(title, fontsize=10, pad=5)
        ax.set_xlabel("UMAP1", fontsize=9)
        ax.set_ylabel("UMAP2", fontsize=9)
        ax.grid(False)
    for ax in axes[1:]:
        ax.set_xlim(axes[0].get_xlim())
        ax.set_ylim(axes[0].get_ylim())
    if entity_handles:
        fig.legend(handles=entity_handles, loc="lower center", bbox_to_anchor=(.5, .005),
                   ncol=len(entity_handles), frameon=False, fontsize=8,
                   handlelength=.8, handletextpad=.4, columnspacing=1.2)
    return fig, np.asarray(axes)


def spatial_panels(coordinates, slice_ids, labels, *, palette, title,
                   slice_order=None, point_size=None, marker="o",
                   display_transforms=None, slice_titles=None, display_bounds=None):
    """Plot all sections in native coordinates using one explicit global palette.

    Use a coordinate DataFrame with spot_id (or a named spot index) to align a
    shuffled label DataFrame by identity. Arrays remain explicitly positional.
    Optional rigid/reflection matrices (2 by 3, including translation) affect
    only plotted copies. Saved coordinates and labels remain unchanged.
    """
    from seareg.plotting import _aligned_labels, _coordinates, spatial_panel

    values = _aligned_labels(coordinates, labels)
    coords = _coordinates(coordinates)
    slices = _text(slice_ids, "sample IDs")
    if len(coords) != len(values) or len(coords) != len(slices):
        raise ValueError("coordinates, labels and sample IDs differ in length")
    order = list(dict.fromkeys(slices)) if slice_order is None else list(_text(slice_order, "slice order", unique=True))
    if set(order) != set(slices):
        raise ValueError("slice order must include each sample exactly once")
    if display_transforms is not None and set(display_transforms) != set(order):
        raise ValueError("display transforms must name every section exactly once")
    cases = []
    for sample in order:
        positions = coords[slices == sample, :2].copy()
        if display_transforms is not None:
            matrix = np.asarray(display_transforms[sample], dtype=float)
            if (matrix.shape != (2, 3) or not np.isfinite(matrix).all()
                    or not np.allclose(matrix[:, :2] @ matrix[:, :2].T, np.eye(2), atol=1e-10)):
                raise ValueError("display transform must preserve spatial distances")
            positions = positions @ matrix[:, :2].T + matrix[:, 2]
        cases.append({"coordinates": positions, "labels": values[slices == sample],
                      "title": (slice_titles or {}).get(sample, sample)})
    if display_bounds is not None:
        bounds = np.asarray(display_bounds, dtype=float)
        if (bounds.shape != (4,) or not np.isfinite(bounds).all()
                or bounds[1] <= bounds[0] or bounds[3] <= bounds[2]):
            raise ValueError("display bounds require increasing x and y limits")
    fig, axes = spatial_panel(cases, figsize=(3.3 * len(order), 3.65), palette=palette,
                              point_size=4 if point_size is None else point_size, marker=marker)
    if display_bounds is not None:
        for ax in axes:
            ax.set_xlim(*bounds[:2]); ax.set_ylim(*bounds[2:])
    legend = axes[-1].get_legend()
    if legend is not None:
        legend.set_title(str(title))
    fig.suptitle(str(title), fontsize=11)
    return fig, axes
