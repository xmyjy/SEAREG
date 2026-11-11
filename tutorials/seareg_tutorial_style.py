"""Reuse example domain names and colors only for an identical partition."""
import json
from pathlib import Path


def tutorial_domain_style(labels, dataset, *, styles_dir=None):
    """Return display labels and a palette without modifying analysis labels.

    The bundled examples retain their established presentation. A new cohort
    or partition keeps its native domain names and the normal plotting colors,
    as does a run without readable optional style files.
    """
    if labels is None:
        return None, None
    styles_dir = Path(styles_dir) if styles_dir is not None else Path(__file__).parent / "styles"
    import pandas as pd
    from matplotlib.colors import is_color_like

    try:
        styles = json.loads((styles_dir / "domain_styles.json").read_text())
        spec = styles.get(dataset) if isinstance(styles, dict) else None
        if not isinstance(spec, dict) or len(labels) != spec.get("spots"):
            return labels, None
        palette = spec.get("palette")
        reference_name = spec.get("labels")
        if (not isinstance(reference_name, str) or not isinstance(palette, dict)
                or not all(isinstance(color, str) and is_color_like(color) for color in palette.values())):
            return labels, None
        reference = pd.read_csv(styles_dir / reference_name, sep="\t", dtype=str,
                                usecols=["spot_id", "domain"])
    except (OSError, ValueError):
        return labels, None

    if (reference.isna().any().any()
            or labels.spot_id.duplicated().any() or reference.spot_id.duplicated().any()
            or set(labels.spot_id) != set(reference.spot_id)):
        return labels, None
    aligned = reference.set_index("spot_id").loc[labels.spot_id]
    native = labels.domain.astype(str)
    mapping = {}
    for domain in native.unique():
        names = aligned.loc[native.to_numpy() == domain, "domain"].unique()
        if len(names) != 1:
            return labels, None
        mapping[domain] = str(names[0])
    if len(set(mapping.values())) != len(mapping) or set(mapping.values()) != set(palette):
        return labels, None
    display = labels.copy()
    display["domain"] = native.map(mapping)
    return display, dict(palette)
